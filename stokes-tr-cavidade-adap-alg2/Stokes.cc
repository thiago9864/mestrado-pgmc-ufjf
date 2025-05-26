
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
#include <ncurses.h>

#include <filesystem>
#include <fstream>
#include <iostream>
namespace CGNS {
using namespace dealii;

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
int limite_nivel_refinamento;
int intervalo_malhas_min;
int intervalo_malhas_max;
int degree_exec;

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
        int passos_entre_refinamentos = 1;

        alpha_pe_map.clear();
        malha_dif_map.clear();
        malha_int_map.clear();
        num_celulas_map.clear();

        printf("\n***Inicia a execucao***\n");

        // Inicializa malha
        // Aqui define o tamanho do domínio, nessa função hyper_cube
        GridGenerator::hyper_cube(triangulation, omega_init, omega_fim);
        triangulation.refine_global(refinamento_inicial);

        // Inicia vetores do detector de oscilação
        int max_cells = std::pow(4, limite_nivel_refinamento);
        direcao_nivel_map = std::vector<int>(max_cells, 0);
        loops_nivel_map = std::vector<int>(max_cells, 0);

        setup_dofs();
        initialize_system();

        // Gera solução pro passo 0
        time = 0.0;
        timestep_number = 0;
        reynolds = 1.0 / visc;
        condicao_inicial();
        old_solution = prev_solution;
        solution = prev_solution;
        ref_solution = prev_solution;
        output_results();

        time_max_number = round(time_end / time_step);
        passo = 0;
        passo_output = 0;

        max_delta_malha = 0;

        printf("Usando o delta t=%f\n", time_step);
        printf("Num de Reynolds=%f\n", reynolds);

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

            printf("\nProblema nao linear com %d células, resolvido no passo de tempo %d (%.2f), com %d iteracoes\n",
                   triangulation.n_active_cells(),
                   timestep_number,
                   time,
                   num_it);

            // Acrescenta um intervalo no salvamento das soluções de forma que
            // um delta t pequeno não gere muitas soluções
            int multPassos = std::round(0.1 / time_step);
            if (multPassos < 1) {
                multPassos = 1;
            }
            multPassos = 10;
            if (passo_output >= multPassos - 1) {
                // Salva solução no passo de tempo
                output_results();
                passo_output = 0;
            } else {
                passo_output++;
            }

            // Estratégia de refinamento
            if (passo >= passos_entre_refinamentos) {
                passos_entre_refinamentos = define_intervalo_entre_refinamentos();

                // Faz o refinamento
                refine_grid();

                // Salva estatisticas relevantes
                grava_estatisticas();

                passo = 0;
            } else {
                // Calcula diferença entre dois passos de tempo
                delta_entre_timesteps = erro_norma_L2(solution, prev_solution);
                // delta_entre_timesteps = erro_residuo(solution);
                // delta_entre_timesteps = erro_kelly(solution);
                residuo_map.insert(std::make_pair(time, delta_entre_timesteps));

                if (timestep_number <= 2 || delta_entre_timesteps < min_delta_entre_timesteps) {
                    min_delta_entre_timesteps = delta_entre_timesteps;
                }
                printf("Aguardando refinamento: passo %d de %d\n", passo, passos_entre_refinamentos);
                passo++;
            }

            printf("delta_entre_timesteps = %.2e, min_delta_entre_timesteps = %.2e\n",
                   delta_entre_timesteps, min_delta_entre_timesteps);
        }

        grava_estatisticas();
        output_results();
        gera_solucoes_eixos();
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

    std::map<double, int> getHashMapNumCelulas() {
        return num_celulas_map;
    }

    std::map<double, double> getHashMapAlphaPe() {
        return alpha_pe_map;
    }

    std::map<double, double> getHashMapMalhaDiff() {
        return malha_dif_map;
    }

    std::map<double, int> getHashMapMalhaInt() {
        return malha_int_map;
    }

    std::map<double, double> getHashMapResiduo() {
        return residuo_map;
    }

    std::string diretorio_stats() {
        int reynolds_int = round(reynolds);
        std::string dir = "stats" + std::to_string(refinamento_inicial);
        dir += "_ordem" + std::to_string(degree_exec);
        dir += "_reynolds" + std::to_string(reynolds_int);

        if (!create_dir(dir)) {
            printf("Erro ao criar diretorio\n");
            exit(3);
        }
        return dir;
    }

    void lerConfiguracoes(std::string arquivo_config) {
        std::string ipName;
        std::ifstream fin(arquivo_config);
        std::string line;
        std::istringstream sin;

        arq_config = arquivo_config;

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
            } else if (line.find("limite_nivel_refinamento") != std::string::npos) {
                sin >> limite_nivel_refinamento;
            } else if (line.find("intervalo_malhas_min") != std::string::npos) {
                sin >> intervalo_malhas_min;
            } else if (line.find("intervalo_malhas_max") != std::string::npos) {
                sin >> intervalo_malhas_max;
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
        printf("limite_nivel_refinamento = %d\n", limite_nivel_refinamento);
        printf("intervalo_malhas_min = %d\n", intervalo_malhas_min);
        printf("intervalo_malhas_max = %d\n", intervalo_malhas_max);

        // exit(2);
        line.clear();
        fin.close();
    }

    void gera_arquivo_saida(double diff) {
        std::string dir_stats = diretorio_stats();

        std::cout << "Tempo de execucao: " << diff / 1000 << " s" << std::endl;

        // Cria arquivo de conclusão
        std::ofstream resultado(dir_stats + "/saida.txt");
        resultado << "Arquivo de configuração usado: " << arq_config << std::endl;
        resultado << "Tempo de execucao: " << diff / 1000 << " s" << std::endl;
        resultado << "Numero de células final: " << getNumCells() << std::endl;
        resultado << "Numero de passos de tempo feitos: " << getPassosTempoSimulado() << std::endl;
        resultado << "Tempo no ultimo passo: " << getTempoSimulado() << "s" << std::endl;
        resultado.close();
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
    BlockVector<double> ref_solution;
    AffineConstraints<double> constraints;
    std::vector<types::global_dof_index> dofs_per_block;
    double time;
    unsigned int timestep_number;
    int limite_celulas = 0;
    double delta_entre_timesteps = 0;
    double min_delta_entre_timesteps = 1e5;
    double min_area_cell = 0;
    double alpha_pe = 0;
    std::map<std::string, int> cell_map;
    double reynolds = 0;
    std::vector<double> alpha_pe_hist;
    std::map<double, double> alpha_pe_map;
    std::map<double, double> malha_dif_map;
    std::map<double, int> malha_int_map;
    std::map<double, double> residuo_map;
    std::map<double, int> num_celulas_map;
    double max_delta_malha = -1e5;
    std::vector<int> direcao_nivel_map;
    std::vector<int> loops_nivel_map;
    std::string arq_config;

    /**
     * Condição de parada do for do tempo
     * @is_aprox_estacionario True considera um erro entre u^n+1 e u^n menor que uma tolerancia,
     * False usa a contagem de passos até o tempo máximo indicado na variavel 'time_end'
     */
    bool condicao_de_parada(bool is_aprox_estacionario) {
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

    double erro_residuo(BlockVector<double> solution_a) {
        std::vector<Point<dim>> vel_points(1);
        vel_points[0] = Point<dim>(0.0, 0.0);
        Quadrature<dim> quadrature(vel_points);
        FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients | update_hessians);
        // std::vector<Vector<double>> local_values(quadrature.size(), Vector<double>(dim + 1));

        const unsigned int n_q_points = quadrature.size();

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)
        const FEValuesExtractors::Scalar pressure(dim);  //(?,?,p)

        // Dados na celula atual
        std::vector<Tensor<1, dim>> solution_atual_u_values(n_q_points);
        std::vector<Tensor<2, dim>> solution_atual_grad_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_lap_u_values(n_q_points);
        // std::vector<double> solution_atual_div_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_grad_p_values(n_q_points);

        Tensor<1, dim> residuo;
        double residuo_medio = 0;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);

            fe_values[velocities].get_function_values(solution_a, solution_atual_u_values);
            fe_values[velocities].get_function_gradients(solution_a, solution_atual_grad_u_values);
            fe_values[velocities].get_function_laplacians(solution_a, solution_atual_lap_u_values);
            //  fe_values[velocities].get_function_divergences(solution_a, solution_atual_div_u_values);
            fe_values[pressure].get_function_gradients(solution_a, solution_atual_grad_p_values);

            residuo = (rho * solution_atual_grad_u_values[0] * solution_atual_u_values[0]  // Termo convectivo
                       - visc * solution_atual_lap_u_values[0]                             // Termo difusivo
                       + solution_atual_grad_p_values[0]                                   // Termo da pressão
            );

            // Calculo da média do resíduo
            residuo_medio += residuo.norm();
        }

        // Termina o calculo da média do resíduo
        return residuo_medio / triangulation.n_active_cells();
    }

    double erro_kelly(BlockVector<double> solution_a) {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)

        KellyErrorEstimator<dim>::estimate(
            dof_handler,
            QGauss<dim - 1>(degree + 1),
            std::map<types::boundary_id, const Function<dim> *>(),
            solution_a,
            estimated_error_per_cell,
            fe.component_mask(velocities));

        double erro_medio = 0;
        int i;
        for (i = 0; i < triangulation.n_active_cells(); i++) {
            erro_medio += estimated_error_per_cell[i];
        }

        // Termina o calculo da média do resíduo
        return erro_medio / triangulation.n_active_cells();
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

        num_celulas_map.insert(std::make_pair(time, triangulation.n_active_cells()));
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

    std::string diretorio_amr() {
        int reynolds_int = round(reynolds);
        std::string dir = "amrdata" + std::to_string(refinamento_inicial);
        dir += "_ordem" + std::to_string(degree);
        dir += "_reynolds" + std::to_string(reynolds_int);

        if (!create_dir(dir)) {
            printf("Erro ao criar diretorio\n");
            exit(3);
        }
        return dir;
    }

    /**
     * Faz a estimativa de erro por gradiente (Kelly) ou resíduo
     * @param estimated_error_per_cell Vetor de estimativas por ordem de chamada do iterator do deal.ii
     * @param usar_kelly_estimator True para estimar erro por gradiente. False para estimar pelo resíduo
     */
    void calcula_funcao_s(Vector<float> &estimated_error_per_cell, bool usar_kelly_estimator) {
        std::vector<Point<dim>> vel_points(1);
        vel_points[0] = Point<dim>(0.0, 0.0);
        Quadrature<dim> quadrature(vel_points);
        FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients | update_hessians);
        // std::vector<Vector<double>> local_values(quadrature.size(), Vector<double>(dim + 1));

        const unsigned int n_q_points = quadrature.size();

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)
        const FEValuesExtractors::Scalar pressure(dim);  //(?,?,p)

        // Dados na celula atual
        std::vector<Tensor<1, dim>> solution_ant_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_u_values(n_q_points);
        std::vector<Tensor<2, dim>> solution_atual_grad_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_lap_u_values(n_q_points);
        // std::vector<double> solution_atual_div_u_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_atual_grad_p_values(n_q_points);

        Tensor<1, dim> residuo;
        Point<dim> p;
        double norma_residuo;
        int indice_celula = 0;

        // Limpa o hashmap pra essa iteração
        cell_map.clear();

        if (usar_kelly_estimator) {
            KellyErrorEstimator<dim>::estimate(
                dof_handler,
                QGauss<dim - 1>(degree + 1),
                std::map<types::boundary_id, const Function<dim> *>(),
                solution,
                estimated_error_per_cell,
                fe.component_mask(velocities));
        }

        // Abre o arquivo CSV de saída
        // Esse arquivo é usado para debug e acompanhamento do refinamento
        std::string dir = diretorio_amr();
        std::ofstream arquivo_csv(dir + "/residuo_t" + std::to_string(timestep_number) + ".csv", std::ios::trunc);
        arquivo_csv << "cell_index,residuo,alpha_pe_ant,pt_x,pt_y" << std::endl;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            if (usar_kelly_estimator == false) {
                fe_values[velocities].get_function_values(prev_solution, solution_ant_u_values);
                fe_values[velocities].get_function_values(solution, solution_atual_u_values);
                fe_values[velocities].get_function_gradients(solution, solution_atual_grad_u_values);
                fe_values[velocities].get_function_laplacians(solution, solution_atual_lap_u_values);
                //  fe_values[velocities].get_function_divergences(solution, solution_atual_div_u_values);
                fe_values[pressure].get_function_gradients(solution, solution_atual_grad_p_values);

                residuo = (rho * (solution_atual_u_values[0] - solution_ant_u_values[0]) / time_step  // Termo no tempo
                           + rho * solution_atual_grad_u_values[0] * solution_atual_u_values[0]       // Termo convectivo
                           - visc * solution_atual_lap_u_values[0]                                    // Termo difusivo
                           + solution_atual_grad_p_values[0]                                          // Termo da pressão
                );
            }

            // Cria um indice numérico mapeado ao indice unico (string) da célula retornado pelo deal.ii
            std::string cell_id = cell->id().to_string();
            cell_map.insert(std::make_pair(cell_id, indice_celula));

            // Armazena a norma simples do resíduo por célula
            if (usar_kelly_estimator == false) {
                norma_residuo = residuo.norm();
                estimated_error_per_cell[indice_celula] = norma_residuo;
            } else {
                // Sai aqui os valores do kelly estimator
                norma_residuo = estimated_error_per_cell[indice_celula];
            }

            // Salva dados de debug e acompanhamento
            p = cell->center();
            arquivo_csv << indice_celula         // cell_index
                        << "," << norma_residuo  // residuo
                        << "," << alpha_pe       // alpha_pe_ant
                        << "," << p[0]           // pt_x
                        << "," << p[1]           // pt_y
                        << std::endl;

            indice_celula++;
        }

        // Fecha o arquivo CSV
        arquivo_csv.close();
    }

    /**
     * Quicksort
     * @param vector Vetor de floats
     */
    void quicksort(Vector<float> &vector, int inicio, int fim) {
        if (inicio < fim) {
            int p = particao(vector, inicio, fim);
            quicksort(vector, inicio, p - 1);
            quicksort(vector, p + 1, fim);
        }
    };

    /**
     * Escolhe um pivo entre o inicio e o fim especificado, depois coloca todos os maiores que ele de um lado e
     * todos os maiores do outro
     * @param vet vetor à ser particionado
     * @param inicio indice que começa a parte a ser particionada
     * @param fim  indice que termina a parte a ser particionada
     * @return a posição que o pivo ficou
     */
    int particao(Vector<float> &vector, int inicio, int fim) {
        int i = inicio - 1;
        float pivo = vector[fim];
        float aux;

        for (int j = inicio; j <= fim - 1; j++) {
            if (vector[j] < pivo) {
                i++;
                aux = vector[i];
                vector[i] = vector[j];
                vector[j] = aux;
            }
        }

        aux = vector[i + 1];
        vector[i + 1] = vector[fim];
        vector[fim] = aux;

        return (i + 1);
    };

    /**
     * Calcula o valor de threshold conforme o artigo 'Adaptive mesh refinement method. Part 1: Automatic
     * thresholding based on a distribution function'
     * @param estimated_error_per_cell Vetor de erro preenchido na ordem de chamada do iterator do deal.ii
     */
    bool calcula_refinamento_fd(Vector<float> estimated_error_per_cell) {
        int num_celulas_diminui_refinamento = 0;
        int num_celulas_aumenta_refinamento = 0;
        int num_celulas_nivel_limite = 0;
        int num_celulas_travadas_por_loop = 0;
        int j, k, n_sk, indice_celula;
        double alpha, func, dist, cell_error;
        double Sm = 0;
        double aux = 0;
        double beta = 2.0;
        int num_celulas = estimated_error_per_cell.size();
        double num_celulas_d = (double)num_celulas;
        Vector<float> est_error_copy(num_celulas);

        // Calcula média do erro
        for (j = 0; j < num_celulas; j++) {
            Sm += estimated_error_per_cell[j];
            // Aproveita e copia o vetor de erros
            est_error_copy[j] = estimated_error_per_cell[j];
        }
        Sm = Sm / num_celulas_d;

        printf("Sm = %f\n", Sm);

        // Ordena a copia do vetor de erros
        quicksort(est_error_copy, 0, num_celulas - 1);

        alpha_pe = Sm;
/*
        // Calcula o threshold
        for (j = 0; j < num_celulas; j++) {
            alpha = Sm * pow(j / num_celulas_d, beta);
            // printf("alpha = %f\n", alpha);
            n_sk = 0;
            for (k = 0; k < num_celulas; k++) {
                if (estimated_error_per_cell[k] > alpha) {
                    n_sk++;
                }
            }
            dist = n_sk / num_celulas_d;
            func = alpha * dist;
            if (func > aux) {
                aux = func;
                alpha_pe = alpha;
            }
        }
*/
        //double alpha_ce = alpha_pe / 4.0;
        double alpha_ce = alpha_pe;
        //int alpha_ind = round(num_celulas * 0.15);
        //double alpha_ce = est_error_copy[alpha_ind];

        printf("alpha_pe = %f\n", alpha_pe);
        printf("alpha_ce = %f\n", alpha_ce);

        alpha_pe_map.insert(std::make_pair(time, alpha_pe));

        indice_celula = 0;
        int direcao_mudanca;
        int lim_loops = 3;  // Limite de loops até travamento da celula
        bool celula_travada;
        bool pode_travar = min_delta_entre_timesteps < 1e-4;

        // Aplica refinamento dado o threshold calculado
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_error = estimated_error_per_cell[indice_celula];
            celula_travada = loops_nivel_map[indice_celula] > lim_loops;
            direcao_mudanca = 0;

            if (cell->level() < limite_nivel_refinamento && cell_error > alpha_pe) {
                if (!pode_travar || !celula_travada) {
                    cell->set_refine_flag();
                    num_celulas_aumenta_refinamento++;
                    direcao_mudanca = 1;
                }
            } else if (cell->level() > refinamento_inicial && cell_error < alpha_ce) {
                if (!pode_travar || !celula_travada) {
                    cell->set_coarsen_flag();
                    num_celulas_diminui_refinamento++;
                    direcao_mudanca = -1;
                }
            }

            if (pode_travar && !celula_travada && direcao_nivel_map[indice_celula] == direcao_mudanca) {
                // Se tiver mudando na mesma direção, reseta o loop
                direcao_nivel_map[indice_celula] = 0;
                loops_nivel_map[indice_celula] = 0;
            } else if (pode_travar && !celula_travada) {
                // Se tiver mudando em direção diferente, registra no contador de loops
                direcao_nivel_map[indice_celula] = direcao_mudanca;
                loops_nivel_map[indice_celula] += 1;
            }

            if (cell->level() >= limite_nivel_refinamento) {
                num_celulas_nivel_limite++;
            }
            if (celula_travada) {
                num_celulas_travadas_por_loop++;
            }

            indice_celula++;
        }

        printf("%d celulas aumentaram o refinamento, %d diminuiram o refinamento\n%d celulas chegaram no nivel limite (%d), %d celulas foram travadas por loop\n",
               num_celulas_aumenta_refinamento,
               num_celulas_diminui_refinamento,
               num_celulas_nivel_limite,
               limite_nivel_refinamento,
               num_celulas_travadas_por_loop);

        // exit(5);
        return num_celulas_aumenta_refinamento > 0 || num_celulas_diminui_refinamento > 0;
    }

    /**
     * Função linear que define de quantos em quantos passos de tempo o refinamento deve ser feito
     * baseado na diferença entre a primeira solução da malha e a ultima antes de refinar pra próxima
     */
    int define_intervalo_entre_refinamentos() {
        double delta_malha = erro_norma_L2(solution, ref_solution);

        if (delta_malha > max_delta_malha) {
            max_delta_malha = delta_malha;
        }

        std::cout << "delta_malha: " << delta_malha << ", max_delta_malha: " << max_delta_malha << std::endl;

        double a = (intervalo_malhas_max - intervalo_malhas_min) / max_delta_malha;
        int intervalo = floor(a * (-delta_malha) + intervalo_malhas_max);

        // Grava log do próximo intervalo
        malha_int_map.insert(std::make_pair(time, intervalo));

        // Grava log do delta da malha anterior
        malha_dif_map.insert(std::make_pair(time, delta_malha));

        return intervalo;
    }

    /**
     * Função que aplica o refinamento na malha
     */
    void refine_grid() {
        bool is_debug = true;
        printf("\n*** Aplica Refinamento ***\n");

        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        std::string python_args;
        const FEValuesExtractors::Vector velocities(0);

        if (is_debug == false) {
            // Usa o estimador de erro sem gerar arquivo CSV com a informação por célula
            // Estimador de erro por gradiente proposto por Kelly et al. (info na documentação do deal.ii)
            KellyErrorEstimator<dim>::estimate(
                dof_handler,
                QGauss<dim - 1>(degree + 1),
                std::map<types::boundary_id, const Function<dim> *>(),
                solution,
                estimated_error_per_cell,
                fe.component_mask(velocities));
            printf("Usando estimador de erro sem gerar arquivo csv\n");
        } else {
            // Usa o estimador de erro gerando arquivo CSV com a informação por célula
            // Segundo parâmetro: True pra estimar por gradiente (Kelly Estimator). False pra estimar pelo resíduo
            calcula_funcao_s(estimated_error_per_cell, true);
            printf("Usando estimador de erro gerando arquivo csv\n");
        }

        // Estrategia de refinamento por limites
        if (calcula_refinamento_fd(estimated_error_per_cell) == false) {
            printf("*** Nenhuma celula pra aumentar ou diminuir o refinamento ***\n");
            return;
        }

        // Aplica as flags de refinamento na malha
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
        ref_solution = tmp;

        printf("*** Refinamento foi aplicado ***\n");
    }

    /**
     * Cria um diretório no mesmo local onde o arquivo Stokes está sendo executado
     * @param path Caminho relativo do diretório
     */
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

    /**
     * Gera arquivo .vtu da solução atual
     */
    void output_results() const {
        int reynolds_int = round(reynolds);
        std::vector<std::string> solution_names(dim, "u");
        solution_names.emplace_back("p");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

        interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.add_data_vector(dof_handler,
                                 solution,
                                 solution_names,
                                 interpretation);

        data_out.build_patches(degree + 1);

        std::string dir = "solution" + std::to_string(refinamento_inicial);
        dir += "_ordem" + std::to_string(degree);
        dir += "_reynolds" + std::to_string(reynolds_int);

        if (create_dir(dir)) {
            std::ofstream output(dir + "/solution_t" + std::to_string(timestep_number) + ".vtu");
            data_out.write_vtu(output);
        } else {
            printf("Erro ao criar diretorio\n");
        }
    }

    /**
     * Gera arquivos csv de linhas nos eixos x e y para comparação com os dados do artigo por Ghia
     */
    void gera_solucoes_eixos() {
        int i;
        int num_pontos = 1000;
        double dx = (omega_fim - omega_init) / num_pontos;
        double dy = dx;

        Point<dim> p_x;
        Point<dim> p_y;

        p_x[0] = 0;
        p_x[1] = 0.5;
        p_y[0] = 0.5;
        p_y[1] = 0;

        int reynolds_int = round(reynolds);
        std::string nome_base = "solution" + std::to_string(refinamento_inicial);
        nome_base += "_ordem" + std::to_string(degree);
        nome_base += "_reynolds" + std::to_string(reynolds_int);

        std::ofstream arq_horizontal(nome_base + "_eixo_x.csv");
        std::ofstream arq_vertical(nome_base + "_eixo_y.csv");

        arq_horizontal << "ux,uy,p,x,y" << std::endl;
        arq_vertical << "ux,uy,p,x,y" << std::endl;

        for (i = 0; i <= num_pontos; i++) {
            Vector<double> tmp_vector(dim + 1);

            VectorTools::point_value(dof_handler, solution, p_x, tmp_vector);
            arq_horizontal << tmp_vector[0]
                           << "," << tmp_vector[1]
                           << "," << tmp_vector[2]
                           << "," << p_x[0]
                           << "," << p_x[1]
                           << std::endl;

            VectorTools::point_value(dof_handler, solution, p_y, tmp_vector);
            arq_vertical << tmp_vector[0]
                         << "," << tmp_vector[1]
                         << "," << tmp_vector[2]
                         << "," << p_y[0]
                         << "," << p_y[1]
                         << std::endl;

            p_x[0] += dx;
            p_y[1] += dy;
        }

        arq_horizontal.close();
        arq_vertical.close();
    }

    void grava_estatisticas() {
        std::string dir_stats = diretorio_stats();

        std::ofstream arq_alpha_pe(dir_stats + "/alpha_pe.csv");
        arq_alpha_pe << "time,alpha_pe" << std::endl;
        for (const auto &[time, alpha_pe] : getHashMapAlphaPe()) {
            arq_alpha_pe << time << "," << alpha_pe << std::endl;
        }
        arq_alpha_pe.close();

        std::ofstream arq_malha_dif(dir_stats + "/malha_dif.csv");
        arq_malha_dif << "time,malha_dif" << std::endl;
        for (const auto &[time, malha_dif] : getHashMapMalhaDiff()) {
            arq_malha_dif << time << "," << malha_dif << std::endl;
        }
        arq_malha_dif.close();

        std::ofstream arq_malha_int(dir_stats + "/malha_int.csv");
        arq_malha_int << "time,malha_int" << std::endl;
        for (const auto &[time, malha_int] : getHashMapMalhaInt()) {
            arq_malha_int << time << "," << malha_int << std::endl;
        }
        arq_malha_int.close();

        std::ofstream arq_residuo(dir_stats + "/residuo.csv");
        arq_residuo << "time,residuo" << std::endl;
        for (const auto &[time, residuo] : getHashMapResiduo()) {
            arq_residuo << time << "," << residuo << std::endl;
        }
        arq_residuo.close();

        std::ofstream arq_num_celulas(dir_stats + "/num_celulas.csv");
        arq_num_celulas << "time,num_celulas" << std::endl;
        for (const auto &[time, num_celulas] : getHashMapNumCelulas()) {
            arq_num_celulas << time << "," << num_celulas << std::endl;
        }
        arq_num_celulas.close();
    }
};
}  // namespace CGNS

int main(int argc, char *argv[]) {
    using namespace dealii;
    using namespace CGNS;

    const int dim = 2;     // 2D
    const int degree = 1;  // Grau dos polinômios usados na aproximação
    degree_exec = degree;

    std::string arquivo_config = "config.txt";

    if (argc < 2) {
        std::cout << "Nenhum arquivo de configuração fornecido. Usando " << arquivo_config << "\n";
    } else if (argc == 2) {
        arquivo_config = argv[1];
        std::cout << "Arquivo de configuração a ser carregado: " << arquivo_config << "\n";
    } else {
        std::cout << "Número de parametros fornecidos é inválido.\nEspera-se ./Stokes ou ./Stokes config.txt" << "\n\n";
        exit(3);
    }

    auto start = std::chrono::steady_clock::now();
    NavierStokesCG<dim> problem(degree);
    problem.lerConfiguracoes(arquivo_config);
    problem.run();
    auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> diff = end - start;
    problem.gera_arquivo_saida(diff.count());

    return 0;
}
