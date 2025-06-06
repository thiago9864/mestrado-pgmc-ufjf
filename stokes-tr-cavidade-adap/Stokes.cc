
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>  // gmsh
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <math.h>

#include <filesystem>
#include <fstream>
#include <iostream>
namespace CGNS {
using namespace dealii;

// Parâmetros do problema
// double visc = 0.0025;
// double rho = 1.0;
// double time_end = 10.0;
// double omega_init = 0.0;
// double omega_fim = 1.0;

// double delta_aprox_estatico = 1e-8;
// unsigned int time_max_number;

double visc;
double rho;
double time_end;
double omega_init;
double omega_fim;
double delta_aprox_estatico;
unsigned int time_max_number;
bool is_aprox_estacionario;
double time_step;
int refinamento_inicial;
int lado_matriz_limite;

/*
 * Classe que contém a representação da função do lado direito da equação de Stokes
 */
template <int dim>
class RightHandSide : public Function<dim> {
   public:
    RightHandSide() : Function<dim>(dim + 1) {}

    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
    }
};

template <int dim>
class BoundaryValues : public Function<dim> {
   public:
    double d = 1e-10;

   public:
    BoundaryValues() : Function<dim>(dim + 1) {}

    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const {
        Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));
        // ux
        values[0] = 0.0;

        // uy
        values[1] = 0.0;

        // p
        values[dim] = 0.0;

        if (abs(p[1] - omega_fim) < d) {
            values[0] = 1.0;
        }
    }
};

/*
 * Classe principal do problema
 */
template <int dim>
class NavierStokesCG {
   public:
    NavierStokesCG(const unsigned int degree)
        // initializes NavierStokesCG::degree
        : degree(degree),
          // initializes NavierStokesCG::fe(grau do pol. p/ (u), num el. u, grau do pol. p/ (p), num el p)
          fe(FE_Q<dim>(degree), dim, FE_Q<dim>(degree), 1),
          dof_handler(triangulation),
          time(0),               // t^n
          timestep_number(0) {}  // O "n" do tempo

    /*
     * Função de execução principal
     * @i Grau de refinamento da malha -> 4^i celulas
     */
    void run() {
        int num_it, passo, passo_output;
        double delta;

        printf("\n***Inicia a execucao***\n");

        // Limita a célula a não ser menor do que uma equivalente em uma grade de 128x128
        // assumindo malha quadrada uniforme.
        limite_celulas = lado_matriz_limite * lado_matriz_limite;
        min_area_cell = pow((omega_fim - omega_init) / sqrt(limite_celulas), 2);

        // Inicializa malha
        // Aqui define o tamanho do domínio, nessa função hyper_cube
        GridGenerator::hyper_cube(triangulation, omega_init, omega_fim);
        triangulation.refine_global(refinamento_inicial);

        setup_dofs();
        initialize_system();

        // Gera solução pro passo 0
        time = 0.0;
        timestep_number = 0;
        condicao_inicial();
        old_solution = prev_solution;
        solution = prev_solution;
        output_results();

        time_max_number = round(time_end / time_step);

        passo = 0;
        passo_output = 0;

        printf("Usando o delta t=%f\n", time_step);
        printf("Num de Reynolds=%f\n", 1.0 / visc);

        if (!is_aprox_estacionario) {
            printf("Execução em %d passos de tempo\n", time_max_number);
        } else {
            printf("Execução aproximadamente estacionária, até o passo atual\n");
            printf("e anterior terem um erro L2 menor que um limite\n\n");
        }

        // Incluir aqui mais um loop pro tempo (etapa 3)
        for (timestep_number = 1; condicao_de_parada(is_aprox_estacionario); timestep_number++) {
            time += time_step;
            num_it = 0;
            passo++;
            passo_output++;

            // Armazena solução em u^{n}
            prev_solution = solution;
            // Inicia solução anterior do método de Picard (u^{n+1 (bar)} = u^{n})
            old_solution = solution;

            // Metodo de Picard
            do {
                system_matrix = 0;
                system_rhs = 0;

                // Calcula solução não linear u^{n+1}
                assemble_system();
                solve();

                // Calcula diferença entre as soluções com a norma L2 discreta
                delta = erro_norma_L2(solution, old_solution);

                // Salva cópia da solução obtida
                // old_solution.block(0) = solution.block(0);
                old_solution = solution;

                num_it++;
                if (num_it > 100) {
                    printf("Número de iteracoes acima do limite de 100\n");
                    printf("Delta alcançado: %f\n", delta);
                    exit(3);
                }
            } while (delta > 1e-6);

            printf("Problema nao linear com %d células, resolvido no passo de tempo %d (%.2f), com %d iteracoes\n",
                   triangulation.n_active_cells(),
                   timestep_number,
                   time,
                   num_it);

            int multPassos = std::round(0.1 / time_step);
            if (multPassos < 1) {
                multPassos = 1;
            }
            if (passo_output >= multPassos) {
                // Salva solução no passo de tempo
                output_results();
                passo_output = 0;
            }

            {{1e-5, 45}, {1e-4, 35}, {1e-3, 25},{1e-2, 15}, {1e-1, 5}}

            // Estratégia de refinamento
            int passos_entre_refinamentos = 15;

            if (min_delta_entre_timesteps > 1e-2) {
                passos_entre_refinamentos = 15 * multPassos;
            } else if (min_delta_entre_timesteps > 1e-3) {
                passos_entre_refinamentos = 25 * multPassos;
            } else if (min_delta_entre_timesteps > 1e-4) {
                passos_entre_refinamentos = 35 * multPassos;
            } else if (min_delta_entre_timesteps > 1e-5) {
                passos_entre_refinamentos = 45 * multPassos;
            } else if (min_delta_entre_timesteps > 1e-6) {
                passos_entre_refinamentos = 55 * multPassos;
            } else if (min_delta_entre_timesteps < 1e-6) {
                passos_entre_refinamentos = 65 * multPassos;
            }

            if (min_delta_entre_timesteps < 1e-5) {
                min_delta_entre_timesteps = 0;
            }

            printf("passos_entre_refinamentos = %d, delta_entre_timesteps = %.2e, min_delta_entre_timesteps = %.2e\n",
                   passos_entre_refinamentos, delta_entre_timesteps, min_delta_entre_timesteps);

            // Refinamento de malha
            if (timestep_number <= 3 || (min_delta_entre_timesteps > 0 && passo >= passos_entre_refinamentos)) {
                passo = 0;
                refine_grid();
            } else {
                printf("Pula Refinamento\n");
            }
        }
    }

    int getNumCells() {
        return triangulation.n_active_cells();
    }

    double getTempoSimulado() {
        return time;
    }

    double getPassosTempoSimulado() {
        return timestep_number;
    }

    void lerConfiguracoes(std::string arquivo_config) {
        std::string ipName;
        std::ifstream fin(arquivo_config);
        std::string line;
        std::istringstream sin;

        while (std::getline(fin, line)) {
            sin.str(line.substr(line.find("=") + 1));
            if (line.find("visc") != std::string::npos) {
                sin >> visc;
            } else if (line.find("rho") != std::string::npos) {
                sin >> rho;
            } else if (line.find("time_end") != std::string::npos) {
                sin >> time_end;
            } else if (line.find("omega_init") != std::string::npos) {
                sin >> omega_init;
            } else if (line.find("omega_fim") != std::string::npos) {
                sin >> omega_fim;
            } else if (line.find("delta_aprox_estatico") != std::string::npos) {
                // std::cout << "delta_aprox_estatico " << sin.str() << ", " << atof(sin.str().c_str()) << std::endl;
                sin >> delta_aprox_estatico;
            } else if (line.find("time_step") != std::string::npos) {
                sin >> time_step;
            } else if (line.find("is_aprox_estacionario") != std::string::npos) {
                sin >> is_aprox_estacionario;
            } else if (line.find("refinamento_inicial") != std::string::npos) {
                sin >> refinamento_inicial;
            } else if (line.find("lado_matriz_limite") != std::string::npos) {
                sin >> lado_matriz_limite;
            }
            sin.clear();
        }
        printf("Executando o problema com as seguintes configurações\n");
        printf("visc = %f\n", visc);
        printf("rho = %f\n", rho);
        printf("time_end = %f\n", time_end);
        printf("omega_init = %f\n", omega_init);
        printf("omega_fim = %f\n", omega_fim);
        printf("delta_aprox_estatico = %.3e\n", delta_aprox_estatico);
        printf("time_step = %f\n", time_step);
        printf("is_aprox_estacionario = %B\n", is_aprox_estacionario);
        printf("lado_matriz_limite = %d\n", lado_matriz_limite);

        // exit(2);
    }

   private:
    const unsigned int degree;
    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double> prev_solution;
    BlockVector<double> solution;
    BlockVector<double> old_solution;
    BlockVector<double> system_rhs;
    AffineConstraints<double> constraints;
    std::vector<types::global_dof_index> dofs_per_block;
    double time;
    unsigned int timestep_number;
    int limite_celulas = 0;
    double delta_entre_timesteps = 0;
    double min_delta_entre_timesteps = 1e5;
    double min_area_cell = 0;
    std::map<std::string, int> cell_map;

    /**
     * Condição de parada do for do tempo
     * @is_aprox_estacionario True considera um erro entre u^n+1 e u^n menor que uma tolerancia,
     * False usa a contagem de passos até o tempo máximo indicado na variavel 'time_end'
     */
    bool condicao_de_parada(bool is_aprox_estacionario) {
        delta_entre_timesteps = erro_norma_L2(solution, prev_solution);
        if (timestep_number <= 2 || delta_entre_timesteps < min_delta_entre_timesteps) {
            min_delta_entre_timesteps = delta_entre_timesteps;
        }
        if (is_aprox_estacionario) {
            return timestep_number <= 2 || delta_entre_timesteps > delta_aprox_estatico;
        } else {
            return timestep_number <= time_max_number;
        }
    }

    /*
     * Calcula erro na norma L2 entre duas soluções
     */
    double erro_norma_L2(BlockVector<double> solution_a, BlockVector<double> solution_b) {
        double erroL2 = 0.0;
        double diff;

        QGauss<dim> quadrature_formula(degree + 1);
        // QGauss<dim - 1> face_quadrature_formula(degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

        const unsigned int n_q_points = quadrature_formula.size();

        // posições na celula atual
        std::vector<Tensor<1, dim>> solution_a_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_b_values(n_q_points);

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);

            fe_values[velocities].get_function_values(solution_a, solution_a_values);
            fe_values[velocities].get_function_values(solution_b, solution_b_values);

            diff = 0;
            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int d = 0; d < dim; ++d) {
                    diff += (pow(solution_b_values[q][d] - solution_a_values[q][d], 2) * fe_values.JxW(q));
                }
            }
            erroL2 += diff;
        }

        return sqrt(erroL2);
    }

    /**
     * Condição inicial do problema
     */
    void condicao_inicial() {
        BoundaryValues<dim> boundary_function;
        VectorTools::project(dof_handler, constraints, QGauss<dim>(degree + 1), boundary_function, prev_solution);
    }

    /**
     * inclui os pontos locais e globais de cada nó
     */
    void setup_dofs() {
        system_matrix.clear();

        dof_handler.distribute_dofs(fe);

        // Configura blocos para velocidade e pressão
        std::vector<unsigned int> block_component(dim + 1, 0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise(dof_handler, block_component);
        dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

        const unsigned int n_u = dofs_per_block[0],
                           n_p = dofs_per_block[1];

        // Condição de contorno
        constraints.clear();

        DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        BoundaryValues<dim> boundary_function;
        FEValuesExtractors::Vector velocity(0);
        VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_function, constraints, fe.component_mask(velocity));

        constraints.close();

        std::cout << "Number of active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << n_u << '+' << n_p << ')' << std::endl
                  << std::endl;
    }

    void initialize_system() {
        BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
        sparsity_pattern.copy_from(dsp);

        system_matrix.reinit(sparsity_pattern);
        system_rhs.reinit(dofs_per_block);
        prev_solution.reinit(dofs_per_block);
        old_solution.reinit(dofs_per_block);
        solution.reinit(dofs_per_block);
    }

    void assemble_system() {
        QGauss<dim> quadrature_formula(degree + 1);
        // QGauss<dim - 1> face_quadrature_formula(degree + 1);
        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients | update_hessians |
                                    update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        // const RightHandSide<dim> right_hand_side;

        std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

        // Aqui entra a inicializaçã do calculo do laplaciano
        std::vector<Tensor<1, dim>> lap_phi_j_u(dofs_per_cell);
        std::vector<Tensor<3, dim>> hess_phi_j_u(dofs_per_cell);

        // posições na celula atual
        std::vector<Tensor<1, dim>> old_solution_values(n_q_points);
        std::vector<Tensor<1, dim>> prev_solution_values(n_q_points);

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)
        const FEValuesExtractors::Scalar pressure(dim);  //(?,?,p)

        const double rho_dt = rho / time_step;

        const double u_norm = old_solution.l2_norm();

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;
            // right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
            //                                   rhs_values);

            fe_values[velocities].get_function_values(old_solution, old_solution_values);
            fe_values[velocities].get_function_values(prev_solution, prev_solution_values);

            // tau
            double h;
            if (dim == 2)
                h = std::sqrt(4. * cell->measure() / M_PI) / degree;
            else if (dim == 3)
                h = std::pow(6. * cell->measure() / M_PI, 1. / 3.) / degree;

            double tau_u = 2.0 * pow(pow(1.0 / time_step, 2) + pow((2.0 * u_norm) / h, 2) + ((4.0 * visc) / (3.0 * h)), -0.5);

            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    // cálculo da hessiana
                    hess_phi_j_u[i] = fe_values[velocities].hessian(i, q);

                    // Cálculo do laplaciano
                    for (int d = 0; d < dim; ++d) {
                        lap_phi_j_u[i][d] = trace(hess_phi_j_u[i][d]);  // lap u
                    }

                    // 1ª equação phi_i_v
                    const Tensor<2, dim> grad_phi_i_v = fe_values[velocities].gradient(i, q);  // grad_v
                    const Tensor<1, dim> phi_i_v = fe_values[velocities].value(i, q);          // v
                    const double div_phi_i_v = fe_values[velocities].divergence(i, q);         // div_v

                    // 2ª equação
                    const double phi_i_q = fe_values[pressure].value(i, q);                  // q
                    const Tensor<1, dim> grad_phi_i_q = fe_values[pressure].gradient(i, q);  // grad_q

                    // P(x) do SUPG
                    const Tensor<1, dim> px_supg = tau_u * (1.0 / rho) * (old_solution_values[q] * grad_phi_i_v - grad_phi_i_q);  // tau * u * grad_v - grad_q

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const Tensor<2, dim> grad_phi_j_u = fe_values[velocities].gradient(j, q);  // grad_u
                        const Tensor<1, dim> phi_j_u = fe_values[velocities].value(j, q);          // u
                        const double div_phi_j_u = fe_values[velocities].divergence(j, q);         // div_u
                        const double phi_j_p = fe_values[pressure].value(j, q);                    // p
                        const Tensor<1, dim> grad_phi_p = fe_values[pressure].gradient(j, q);      // grad_p

                        // O indice da matriz local determina quem, a respeito de i,j vai ser
                        // a solução ou função teste. A primeira componente do local_matrix
                        // é a função teste (i), e a segunda é a função solução (j)
                        local_matrix(i, j) += (visc * scalar_product(grad_phi_j_u, grad_phi_i_v)        // A:  mu * (grad_u, grad_v)
                                               + rho_dt * phi_j_u * phi_i_v                             // M:  (rho/dt) * (u, v)
                                               + rho * grad_phi_j_u * old_solution_values[q] * phi_i_v  // C:  rho * u_old * grad_u * v // Termo convectivo
                                               - phi_j_p * div_phi_i_v                                  // B1: -(p,div_v)
                                               - phi_i_q * div_phi_j_u                                  // B2: -(q,div_u)
                                               - 1e-8 * phi_j_p * phi_i_q                               // -1e-8 * (p,q)   -> Isso elimina os valores zero na diagonal principal
                                               - visc * (lap_phi_j_u[i] * px_supg)                      // S_A: -mu * lap_u * p_x
                                               + rho_dt * phi_j_u * px_supg                             // S_M: (rho/dt) * u * p_x
                                               + grad_phi_j_u * old_solution_values[q] * px_supg        // S_C: rho * u_old * grad_u * p_x
                                               + grad_phi_p * px_supg)                                  // S_B1: grad_p * p_x
                                              * fe_values.JxW(q);                                       // Esse J e W já incluem a função peso e W da integração
                    }

                    local_rhs(i) += (rho_dt * prev_solution_values[q] * phi_i_v    // F:   (rho/dt) * (u_prev, v)
                                     + rho_dt * prev_solution_values[q] * px_supg  // S_f: (rho/dt) * (u_prev, p_x)
                                     ) *
                                    fe_values.JxW(q);
                }
            }
            cell->get_dof_indices(local_dof_indices);

            // Inclui as condições de contorno na matriz global e fonte global nessa
            // variavel constraints, que saem da matriz global e fonte as linhas e colunas
            // relacionadas as condições de contorno
            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
        }
    }

    void solve() {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(solution, system_rhs);

        // Impõe as condições de contorno no vetor solução que ficaram guardadas nele
        // na fase de construção
        constraints.distribute(solution);
    }

    void residuo_L2(BlockVector<double> solution_atual, BlockVector<double> solution_ant, Vector<float> &estimated_error_per_cell, std::string &python_args) {
        double res_time, res_conv, res_dif, res_pre, aux_res, res_eq;
        double aux_time, aux_conv, aux_dif, aux_pre;
        int i;

        std::vector<std::vector<double>> componentes(
            estimated_error_per_cell.size(),
            std::vector<double>(5, 0.0));

        std::vector<std::string> mesh_info(estimated_error_per_cell.size());

        Vector<int> cell_indexes(estimated_error_per_cell.size());
        QGauss<dim> quadrature_formula(degree + 1);
        // QGauss<dim - 1> face_quadrature_formula(degree + 1);
        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients | update_hessians |
                                    update_quadrature_points | update_JxW_values);

        const unsigned int n_q_points = quadrature_formula.size();

        // posições na celula atual
        std::vector<Tensor<1, dim>> solution_ant_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_u_values(n_q_points);
        std::vector<Tensor<2, dim>> solution_atual_grad_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_lap_u_values(n_q_points);
        // std::vector<double> solution_atual_div_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_grad_p_values(n_q_points);

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)
        const FEValuesExtractors::Scalar pressure(dim);  //(?,?,p)

        Tensor<1, dim> q_time;
        Tensor<1, dim> q_conv;
        Tensor<1, dim> q_dif;
        Tensor<1, dim> q_pre;
        i = 0;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);

            fe_values[velocities].get_function_values(solution_ant, solution_ant_u_values);
            fe_values[velocities].get_function_values(solution_atual, solution_atual_u_values);
            fe_values[velocities].get_function_gradients(solution_atual, solution_atual_grad_u_values);
            fe_values[velocities].get_function_laplacians(solution_atual, solution_atual_lap_u_values);
            //  fe_values[velocities].get_function_divergences(solution_atual, solution_atual_div_u_values);
            fe_values[pressure].get_function_gradients(solution_atual, solution_atual_grad_p_values);

            res_time = 0;
            res_conv = 0;
            res_dif = 0;
            res_pre = 0;
            res_eq = 0;

            for (unsigned int q = 0; q < n_q_points; ++q) {
                q_time = rho * (solution_atual_u_values[q] - solution_ant_u_values[q]) / time_step;  // Termo no tempo
                q_conv = rho * solution_atual_grad_u_values[q] * solution_atual_u_values[q];         // Termo convectivo
                q_dif = visc * solution_atual_lap_u_values[q];                                       // Termo difusivo
                q_pre = solution_atual_grad_p_values[q];                                             // Termo da pressão

                aux_time = 0;
                aux_conv = 0;
                aux_dif = 0;
                aux_pre = 0;
                aux_res = 0;

                for (unsigned int d = 0; d < dim; ++d) {
                    aux_time += q_time[d];
                    aux_conv += q_conv[d];
                    aux_dif += q_dif[d];
                    aux_pre += q_pre[d];
                    // aux_time += pow(q_time[d], 2) * fe_values.JxW(q);
                    // aux_conv += pow(q_conv[d], 2) * fe_values.JxW(q);
                    // aux_dif += pow(q_dif[d], 2) * fe_values.JxW(q);
                    // aux_pre += pow(q_pre[d], 2) * fe_values.JxW(q);
                    aux_res += pow(q_time[d] + q_conv[d] - q_dif[d] + q_pre[d], 2) * fe_values.JxW(q);
                }

                res_time += aux_time;
                res_conv += aux_conv;
                res_dif += aux_dif;
                res_pre += aux_pre;
                res_eq += sqrt(aux_res);
            }

            cell_indexes[i] = cell->index();
            estimated_error_per_cell[i] = res_eq / n_q_points;
            componentes[i][0] = res_time / n_q_points;
            componentes[i][1] = res_conv / n_q_points;
            componentes[i][2] = res_dif / n_q_points;
            componentes[i][3] = res_pre / n_q_points;
            componentes[i][4] = estimated_error_per_cell[i];
            // printf("%f ", estimated_error_per_cell[i]);
            //  if (i % 10 == 0) {
            //      printf("\n");
            //  }
            mesh_info[i] = "";
            for (const auto ci : cell->vertex_indices()) {
                Point<2> &p = cell->vertex(ci);
                // std::cout << "teste 0: " << cell->vertex_dof_index(i, 0) << ", 1: " << cell->vertex_dof_index(i, 1) << std::endl;
                // std::cout << "cell-id: " << cell->index() << ", vertex-ind:" << i << ", x: " << p[0] << ", y: " << p[1] << std::endl;

                mesh_info[i] += std::to_string(ci)                                     // indice local do vertice
                                + "|" + std::to_string(cell->vertex_dof_index(ci, 0))  // dof index x
                                + "|" + std::to_string(cell->vertex_dof_index(ci, 1))  // dof index y
                                + "|" + std::to_string(p[0])                           // posicao x
                                + "|" + std::to_string(p[1])                           // posicao y
                                + ":";                                                 // separador de vertices
            }
            // std::cout << "mesh_info -> " << mesh_info[i] << std::endl;

            i++;
        }
        printf("\n");

        float min = estimated_error_per_cell[0];
        float max = estimated_error_per_cell[0];
        for (i = 1; i < estimated_error_per_cell.size(); i++) {
            if (estimated_error_per_cell[i] < min) {
                min = estimated_error_per_cell[i];
            }
            if (estimated_error_per_cell[i] > max) {
                max = estimated_error_per_cell[i];
            }
        }

        printf("estimated_error_per_cell max=%f, min=%f\n", max, min);

        // Cria arquivo de saida do residuo
        std::string dir = "amrdata" + std::to_string(refinamento_inicial) + "_ordem" + std::to_string(degree);

        if (!create_dir(dir)) {
            printf("Erro ao criar diretorio\n");
            exit(3);
        }

        std::ofstream resultado(dir + "/residuo_t" + std::to_string(timestep_number) + ".csv");

        resultado << "cell_index,tempo,conveccao,difusao,pressao,residuo,mesh_info" << std::endl;
        python_args = "";
        for (int i = 0; i < cell_indexes.size(); i++) {
            resultado << cell_indexes[i]
                      << "," << componentes[i][0]
                      << "," << componentes[i][1]
                      << "," << componentes[i][2]
                      << "," << componentes[i][3]
                      << "," << componentes[i][4]
                      << "," << mesh_info[i]
                      << std::endl;
            python_args += std::to_string(cell_indexes[i])            // indice da celula
                           + "," + std::to_string(componentes[i][0])  // tempo
                           + "," + std::to_string(componentes[i][1])  // convecção
                           + "," + std::to_string(componentes[i][2])  // difusão
                           + "," + std::to_string(componentes[i][3])  // pressão
                           + "," + std::to_string(componentes[i][4])  // residuo
                           + "|";                                     // separador
        }

        resultado.close();

        // Escreve como svg a malha final
        std::ofstream mesh_svg_file(dir + "/mesh_t" + std::to_string(timestep_number) + ".svg");
        GridOutFlags::Svg svg_flags;
        GridOut grid_out;

        svg_flags.label_cell_index = true;
        grid_out.set_flags(svg_flags);
        grid_out.write_svg(triangulation, mesh_svg_file);
    }

    std::string diretorio_amr() {
        std::string dir = "amrdata" + std::to_string(refinamento_inicial) + "_ordem" + std::to_string(degree);
        if (!create_dir(dir)) {
            printf("Erro ao criar diretorio\n");
            exit(3);
        }
        return dir;
    }

    void calcula_componentes_e_residuo(Vector<float> &estimated_error_per_cell) {
        QGauss<dim> quadrature_formula(degree + 1);
        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients | update_hessians | update_quadrature_points | update_JxW_values);

        const unsigned int n_q_points = quadrature_formula.size();

        if (n_q_points != 9) {
            std::cout << "Só funciona com celulas 2D com aproximação Q2-Q2!" << std::endl;
            exit(1);
        }

        // Testa componentes com elementos finitos
        std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
        Vector<double> local_dof_values(fe.n_dofs_per_cell());
        Vector<double> local_dof_values_ant(fe.n_dofs_per_cell());

        // posições na celula atual
        std::vector<Tensor<1, dim>> solution_ant_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_u_values(n_q_points);
        std::vector<Tensor<2, dim>> solution_atual_grad_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_lap_u_values(n_q_points);
        // std::vector<double> solution_atual_div_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_grad_p_values(n_q_points);

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)
        const FEValuesExtractors::Scalar pressure(dim);  //(?,?,p)

        // Armazena as componentes
        std::vector<double> componentes(8);

        double residuo_x, residuo_y;
        int indice_celula = 0;
        int q = 8;  // Na celula 2d Q2-Q2 o ultimo ponto da quadratura é o do centro

        // Limpa o hashmap pra essa iteração
        cell_map.clear();

        // Abre o arquivo CSV de saída
        std::string dir = diretorio_amr();
        std::ofstream arquivo_csv(dir + "/residuo_t" + std::to_string(timestep_number) + ".csv", std::ios::trunc);
        arquivo_csv << "cell_index,ace_x,ace_y,con_x,con_y,dif_x,dif_y,pre_x,pre_y,res_x,res_y,pt_x,pt_y" << std::endl;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);

            // std::cout << "cell->id(): " << cell->id().to_string() << std::endl;
            //  int cellId = cell->id().to_binary();

            fe_values[velocities]
                .get_function_values(prev_solution, solution_ant_u_values);
            fe_values[velocities].get_function_values(solution, solution_atual_u_values);
            fe_values[velocities].get_function_gradients(solution, solution_atual_grad_u_values);
            fe_values[velocities].get_function_laplacians(solution, solution_atual_lap_u_values);
            //  fe_values[velocities].get_function_divergences(solution, solution_atual_div_u_values);
            fe_values[pressure].get_function_gradients(solution, solution_atual_grad_p_values);

            // Supondo que a célula seja 2D e Q2-Q2, vai usar só o ponto da quadratura no centro da celula
            for (unsigned int d = 0; d < dim; ++d) {
                componentes[d] = rho * (solution_atual_u_values[q][d] - solution_ant_u_values[q][d]) / time_step;  // Termo no tempo
                componentes[d + 2] = rho * (solution_atual_grad_u_values[q] * solution_atual_u_values[q])[d];      // Termo convectivo
                componentes[d + 4] = visc * solution_atual_lap_u_values[q][d];                                     // Termo difusivo
                componentes[d + 6] = solution_atual_grad_p_values[q][d];                                           // Termo da pressão
            }

            // calcula residuo
            residuo_x = componentes[0] + componentes[2] - componentes[4] + componentes[6];
            residuo_y = componentes[1] + componentes[3] - componentes[5] + componentes[7];

            // Pra métrica de refinamento padrão, é retornado esse array com a magnitude do erro
            estimated_error_per_cell[indice_celula] = sqrt(pow(residuo_x, 2) + pow(residuo_y, 2));

            // ponto central da célula
            Point<dim> p = cell->center();
            std::string cell_id = cell->id().to_string();
            cell_map.insert(std::make_pair(cell_id, indice_celula));

            arquivo_csv << indice_celula          // cell_index
                        << "," << componentes[0]  // ace_x
                        << "," << componentes[1]  // ace_y
                        << "," << componentes[2]  // con_x
                        << "," << componentes[3]  // con_y
                        << "," << componentes[4]  // dif_x
                        << "," << componentes[5]  // dif_y
                        << "," << componentes[6]  // pre_x
                        << "," << componentes[7]  // pre_y
                        << "," << residuo_x       // res_x
                        << "," << residuo_y       // res_y
                        << "," << p[0]            // pt_x
                        << "," << p[1]            // pt_y
                        << std::endl;

            indice_celula++;
        }

        // Fecha o arquivo CSV
        arquivo_csv.close();
    }

    void calcula_refinamento_python() {
        // Comando a ser executado
        std::string dir = diretorio_amr();
        std::string cmd = "python3 refinamento.py " + dir + " " + std::to_string(timestep_number);
        // std::cout << "cmd: " << cmd << std::endl;

        // Executa o comando
        std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            std::cout << "Error: popen failed!" << std::endl;
            exit(5);
        }

        // Leitura do stdout do python
        char buffer[128];
        std::string result = "";
        while (!feof(pipe.get())) {
            if (fgets(buffer, 128, pipe.get()) != nullptr) {
                result += buffer;
            }
        }
        // std::cout << "resultado: " << result << "\n";

        std::istringstream f(result.c_str());
        std::string line;
        std::istringstream sin;
        std::istringstream sin2;
        std::istringstream sin3;

        // Inicia vector onde o indice é o mesmo da célula
        std::vector<int> flag_refinamento(triangulation.n_active_cells());
        for (int i = 0; i < triangulation.n_active_cells(); i++) {
            flag_refinamento[i] = 0;
        }

        // Interpreta stdout do python
        int indice_celula;
        int delta_celula;
        bool erro = true;
        while (std::getline(f, line)) {
            sin.str(line.substr(line.find("=") + 1));
            if (line.find("start") != std::string::npos) {
                erro = false;
            } else if (!erro && line.find("cell_index") != std::string::npos) {
                std::string teste1;
                sin >> teste1;
                int sep = teste1.find(",");
                sin2.str(teste1.substr(sep + 1));
                sin2 >> delta_celula;
                sin3.str(teste1.substr(0, sep));
                sin3 >> indice_celula;
                // std::cout << "teste1: " << teste1 << ", indice_celula: " << indice_celula << ", delta_celula: " << delta_celula << std::endl;
                flag_refinamento[indice_celula] = delta_celula;
            }
            sin.clear();
            sin2.clear();
            sin3.clear();
        }
        if (erro) {
            std::cout << "Ocorreu um erro ao tentar interpretar retorno do script python" << std::endl;
            exit(1);
        }

        // Marca pra refinamento
        int num_celulas_diminui_refinamento = 0;
        int num_celulas_aumenta_refinamento = 0;
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            std::string cell_id = cell->id().to_string();

            if (auto busca = cell_map.find(cell_id); busca != cell_map.end()) {
                indice_celula = busca->second;
            } else {
                std::cout << "Celula '" << cell_id << "' não encontrada no mapeamento!\n";
                exit(5);
            }

            int delta = flag_refinamento[indice_celula];

            if (delta > 0) {
                cell->set_refine_flag();
                num_celulas_aumenta_refinamento++;
            } else if (delta < 0) {
                cell->set_coarsen_flag();
                num_celulas_diminui_refinamento++;
            }
        }

        printf("%d celulas aumentaram o refinamento e %d diminuiram o refinamento refinamento\n",
               num_celulas_aumenta_refinamento,
               num_celulas_diminui_refinamento);
    }

    void calcula_refinamento_limite(Vector<float> estimated_error_per_cell) {
        int i = 0;
        bool is_tam_minimo;
        int num_celulas_diminui_refinamento = 0;
        int num_celulas_aumenta_refinamento = 0;
        float max_error = estimated_error_per_cell[0];
        float min_error = estimated_error_per_cell[0];

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (estimated_error_per_cell[i] > max_error) {
                max_error = estimated_error_per_cell[i];
            }
            if (estimated_error_per_cell[i] < min_error) {
                min_error = estimated_error_per_cell[i];
            }
            i++;
        }

        float threshold = ((max_error - min_error) * 0.3) + min_error;

        printf("threshold: %f\n", threshold);

        i = 0;
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            is_tam_minimo = cell->measure() - min_area_cell < min_area_cell * 0.1;
            if (is_tam_minimo == false && estimated_error_per_cell[i] > threshold) {
                cell->set_refine_flag();
                num_celulas_aumenta_refinamento++;
            } else if (timestep_number > 3 && estimated_error_per_cell[i] < 0.01) {
                cell->set_coarsen_flag();
                num_celulas_diminui_refinamento++;
            }
            i++;
        }

        printf("%d celulas aumentaram o refinamento e %d diminuiram o refinamento refinamento. Maior erro: %f, Menor erro: %f\n",
               num_celulas_aumenta_refinamento,
               num_celulas_diminui_refinamento,
               max_error,
               min_error);
    }
    /**
     * Estratégia de refinamento 2D: Aplica o refinamento padrão, que dobra o número de células
     * a cada passo de refinamento, mas impede que células abaixo de um tamanho mínimo sejam
     * refinadas. Células a serem mescladas não são interrompidas
     */
    void calcula_refinamento_2D(Vector<float> estimated_error_per_cell) {
        // Estratégia de refinamento padrão
        double fracao_dividir = 0.3;
        double fracao_mesclar = 0.03;
        int num_celulas_ajustadas = 0;

        // Estratégia pronta do dealii
        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                        estimated_error_per_cell,
                                                        fracao_dividir,   // Dividir células com maior erro
                                                        fracao_mesclar,   // Mesclar células com menor erro
                                                        limite_celulas);  // Número máximo de células na malha

        // Desmarca refinamento de células com area menor que o limite
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (cell->measure() - min_area_cell < min_area_cell * 0.1) {
                cell->clear_refine_flag();
                num_celulas_ajustadas++;
            }
        }

        printf("%d celulas já chegaram no tamanho mínimo\n", num_celulas_ajustadas);
    }

    void refine_grid() {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        std::string python_args;
        const FEValuesExtractors::Vector velocity(0);

        // KellyErrorEstimator<dim>::estimate(
        //     dof_handler,
        //     QGauss<dim - 1>(degree + 1),
        //     std::map<types::boundary_id, const Function<dim> *>(),
        //     solution,
        //     estimated_error_per_cell,
        //     fe.component_mask(velocity));

        // Estima o erro pelo resíduo
        calcula_componentes_e_residuo(estimated_error_per_cell);

        printf("\n*** Aplica Refinamento ***\n");

        // Estratégia de refinamento por script python externo
        // calcula_refinamento_python();
        // exit(4);
        // Estratégia de refinamento do deal.ii com limite de células
        // calcula_refinamento_2D(estimated_error_per_cell);

        // Estrategia de refinamento por limites
        calcula_refinamento_limite(estimated_error_per_cell);

        triangulation.prepare_coarsening_and_refinement();
        SolutionTransfer<dim, BlockVector<double>> solution_transfer(dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(solution);
        triangulation.execute_coarsening_and_refinement();

        // First the DoFHandler is set up and constraints are generated. Then we
        // create a temporary BlockVector <code>tmp</code>, whose size is
        // according with the solution on the new mesh.
        setup_dofs();

        BlockVector<double> tmp(dofs_per_block);

        // Transfer solution from coarse to fine mesh and apply boundary value
        // constraints to the new transferred solution. Note that 'solution'
        // is still a vector corresponding to the old mesh.
        solution_transfer.interpolate(solution, tmp);
        constraints.distribute(tmp);

        // Finally set up matrix and vectors and set 'solution' to the
        // interpolated data.
        initialize_system();
        solution = tmp;
    }

    bool create_dir(std::string path) const {
        try {
            if (std::filesystem::is_directory(path) && std::filesystem::exists(path)) {  // Check if src folder exists
                return true;
            }
            return std::filesystem::create_directory(path);
        } catch (std::exception &e) {
            std::cerr << e.what() << std::endl;
            return false;
        }
        return true;
    }

    void output_results() const {
        std::vector<std::string> solution_names(dim, "u");
        solution_names.emplace_back("p");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation(dim,
                           DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.add_data_vector(dof_handler,
                                 solution,
                                 solution_names,
                                 interpretation);
        data_out.build_patches(degree + 1);

        std::string dir = "solution" + std::to_string(refinamento_inicial) + "_ordem" + std::to_string(degree);

        if (create_dir(dir)) {
            std::ofstream output(dir + "/solution_t" + std::to_string(timestep_number) + ".vtu");
            data_out.write_vtu(output);
        } else {
            printf("Erro ao criar diretorio\n");
        }
    }

    void print_values() {
        std::vector<Point<dim>> vel_points(1);
        vel_points[0] = Point<dim>(0.0, 0.0);
        Quadrature<dim> quadrature(vel_points);
        FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients);
        std::vector<Vector<double>> local_values(quadrature.size(), Vector<double>(dim + 1));

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            fe_values.get_function_values(solution, local_values);
            //            fe_values.get_function_gradients(solution_local, grad);

            // uy na direção em y=0.5
            if (cell->vertex(0)(1) == 0.5) {
                printf("%f   %f \n", cell->vertex(0)(0), local_values[0](1));
            }
        }
    }
};
}  // namespace CGNS

int main(int argc, char *argv[]) {
    using namespace dealii;
    using namespace CGNS;

    const int dim = 2;     // 2D
    const int degree = 2;  // Grau dos polinômios usados na aproximação

    std::string arquivo_config = "config.txt";
    std::string arquivo_saida = "saida.txt";

    if (argc < 2) {
        std::cout << "Nenhum arquivo de configuração fornecido. Usando " << arquivo_config << "\n";
        std::cout << "Nenhum arquivo de saída fornecido. Usando " << arquivo_saida << "\n\n";
    } else if (argc == 2) {
        arquivo_config = argv[1];
        std::cout << "Arquivo de configuração a ser carregado: " << arquivo_config << "\n";
        std::cout << "Nenhum arquivo de saída fornecido. Usando " << arquivo_saida << "\n\n";
    } else if (argc == 3) {
        arquivo_config = argv[1];
        arquivo_saida = argv[2];
        std::cout << "Arquivo de configuração a ser carregado: " << arquivo_config << "\n";
        std::cout << "Arquivo de saída a ser gerado: " << arquivo_saida << "\n\n";
    } else {
        std::cout << "Número de parametros fornecidos é inválido.\nEspera-se ./Stokes, ./Stokes config.txt ou ./Stokes config.txt saida.txt" << "\n\n";
        exit(3);
    }

    auto start = std::chrono::steady_clock::now();
    NavierStokesCG<dim> problem(degree);
    problem.lerConfiguracoes(arquivo_config);
    problem.run();
    auto end = std::chrono::steady_clock::now();

    // Store the time difference between start and end
    auto diff = end - start;

    std::cout << "Tempo de execucao: " << std::chrono::duration<double, std::milli>(diff).count() / 1000 << " s" << std::endl;

    // Cria arquivo de conclusão
    std::ofstream resultado(arquivo_saida);

    resultado << "Arquivo de configuração usado: " << arquivo_config << std::endl;
    resultado << "Tempo de execucao: " << std::chrono::duration<double, std::milli>(diff).count() / 1000 << " s" << std::endl;
    resultado << "Numero de células final: " << problem.getNumCells() << std::endl;
    resultado << "Numero de passos de tempo feitos: " << problem.getPassosTempoSimulado() << std::endl;
    resultado << "Tempo no ultimo passo: " << problem.getTempoSimulado() << "s" << std::endl;

    resultado.close();

    return 0;
}
