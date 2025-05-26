
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
#include <deal.II/numerics/matrix_tools.h>
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
// double time_step;

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
        values[0] = 0.0;    // u_x
        values[1] = 0.0;    // u_y
        values[dim] = 0.0;  // p

        if (abs(p[1] - omega_fim) < d) {
            values[0] = 1.0;  // u_x
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
     * @i Grau de refinamento da malha -> 2^i elementos
     * @convergence_table Tabela de convergência
     */
    void run() {
        int num_it;
        double delta;
        int passo_output = 0;

        // Inicializa malha
        make_grid_and_dofs();

        // Gera solução pro passo 0
        reynolds = 1.0 / visc;
        time = 0.0;
        timestep_number = 0;
        inicia_condicao_contorno();
        condicao_inicial();
        // constraints.distribute(solution);
        old_solution = prev_solution;
        solution = prev_solution;
        output_results();

        // Define o tamanho do passo de tempo proporcional ao valor de h
        time_step = 0.1;
        time_max_number = round(time_end / time_step);
        

        printf("Usando o delta t=%f\n", time_step);
        printf("Num de Reynolds=%f\n", 1.0 / visc);
        printf("Execução em %d passos de tempo\n", time_max_number);

        // Incluir aqui mais um loop pro tempo (etapa 3)
        for (timestep_number = 1; condicao_de_parada(is_aprox_estacionario); timestep_number++) {
            time += time_step;
            num_it = 0;

            // Armazena solução em u^{n}
            prev_solution = solution;

            // old_solution = 0;
            inicia_condicao_contorno();

            // printf("dt=%f, t=%f\n", time_step, time);

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

            printf("Problema com %d células, resolvido no passo de tempo %d (%.2f s), com %d iteracoes\n",
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
            if (passo_output >= multPassos - 1) {
                // Salva solução no passo de tempo
                output_results();
                passo_output = 0;
            } else {
                passo_output++;
            }
        }

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
        printf("refinamento_inicial = %d\n", refinamento_inicial);

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
    double time;
    unsigned int timestep_number;
    unsigned int time_max_number;
    double reynolds = 0;

    /**
     * Condição de parada do for do tempo
     * @is_aprox_estacionario True considera um erro entre u^n+1 e u^n menor que uma tolerancia,
     * False usa a contagem de passos até o tempo máximo indicado na variavel 'time_end'
     */
    bool condicao_de_parada(bool is_aprox_estacionario) {
        if (is_aprox_estacionario) {
            double delta = erro_norma_L2(solution, prev_solution);
            // printf("Timestep: %d, Delta: %f\n", timestep_number, delta);
            return timestep_number <= 2 || delta > delta_aprox_estatico;
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
        QGauss<dim - 1> face_quadrature_formula(degree + 1);
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

    void condicao_inicial() {
        BoundaryValues<dim> boundary_function;
        VectorTools::project(dof_handler, constraints, QGauss<dim>(degree + 1), boundary_function, prev_solution);
    }

    /**
     * Inicializa a malha e inclui os pontos locais e globais de cada nó
     */
    void make_grid_and_dofs() {
        // Aqui define o tamanho do domínio, nessa função hyper_cube
        GridGenerator::hyper_cube(triangulation, omega_init, omega_fim);
        triangulation.refine_global(refinamento_inicial);
        dof_handler.distribute_dofs(fe);

        std::vector<unsigned int> block_component(dim + 1, 0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise(dof_handler, block_component);
        const std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

        const unsigned int n_u = dofs_per_block[0],
                           n_p = dofs_per_block[1];

        std::cout << "Number of active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << n_u << '+' << n_p << ')' << std::endl
                  << std::endl;

        prev_solution.reinit(dofs_per_block);
        solution.reinit(dofs_per_block);
        system_rhs.reinit(dofs_per_block);
        old_solution.reinit(dofs_per_block);

        BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
        DoFTools::make_sparsity_pattern(dof_handler, dsp);

        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        // Escreve como svg
        // std::ofstream mesh_svg_file("mesh" + std::to_string(i) + ".svg");
        // GridOut().write_svg(triangulation, mesh_svg_file);
    }

    /**
     * Inicia condição de contorno com o tempo atual do problema
     */
    void inicia_condicao_contorno() {
        constraints.clear();
        BoundaryValues<dim> boundary_function;
        FEValuesExtractors::Vector velocity(0);
        VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_function, constraints, fe.component_mask(velocity));
        constraints.close();
    }

    void assemble_system() {
        QGauss<dim> quadrature_formula(degree + 1);
        QGauss<dim - 1> face_quadrature_formula(degree + 1);
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
        double h;
        double tau_u;
        Tensor<2, dim> grad_phi_i_v;
        Tensor<1, dim> phi_i_v;
        double div_phi_i_v;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;

            fe_values[velocities].get_function_values(old_solution, old_solution_values);
            fe_values[velocities].get_function_values(prev_solution, prev_solution_values);

            // tau
            if (dim == 2)
                h = std::sqrt(4. * cell->measure() / M_PI) / degree;
            else if (dim == 3)
                h = std::pow(6. * cell->measure() / M_PI, 1. / 3.) / degree;

            tau_u = pow(pow(1.0 / time_step, 2) + pow((2.0 * u_norm) / h, 2) + ((4.0 * visc) / (3.0 * h)), -0.5);

            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    // cálculo da hessiana
                    hess_phi_j_u[i] = fe_values[velocities].hessian(i, q);

                    // Cálculo do laplaciano
                    for (int d = 0; d < dim; ++d) {
                        lap_phi_j_u[i][d] = trace(hess_phi_j_u[i][d]);  // lap u
                    }

                    // 1ª equação phi_i_v
                    grad_phi_i_v = fe_values[velocities].gradient(i, q);   // grad_v
                    phi_i_v = fe_values[velocities].value(i, q);           // v
                    div_phi_i_v = fe_values[velocities].divergence(i, q);  // div_v

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
                                                   system_matrix, system_rhs);
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
            interpretation(dim,
                           DataComponentInterpretation::component_is_part_of_vector);
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
};
}  // namespace CGNS

int main(int argc, char *argv[]) {
    using namespace dealii;
    using namespace CGNS;

    const int dim = 2;     // 2D
    const int degree = 3;  // Usando polinomios de ordem 1

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