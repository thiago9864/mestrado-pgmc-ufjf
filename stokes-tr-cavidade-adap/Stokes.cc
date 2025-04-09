
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
double visc = 0.0025;
double rho = 1.0;
double time_end = 10.0;
double omega_init = 0.0;
double omega_fim = 1.0;

unsigned int time_max_number;

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
     * @i Grau de refinamento da malha -> 2^(dim*i) elementos
     */
    void run(int i) {
        int num_it;
        double delta;

        // Inicializa malha
        // Aqui define o tamanho do domínio, nessa função hyper_cube
        GridGenerator::hyper_cube(triangulation, omega_init, omega_fim);
        triangulation.refine_global(i);

        setup_dofs();
        initialize_system();

        // Gera solução pro passo 0
        time = 0.0;
        timestep_number = 0;
        condicao_inicial();
        old_solution = prev_solution;
        solution = prev_solution;
        output_results(i);

        time_step = 0.1;
        time_max_number = round(time_end / time_step);

        printf("Usando o delta t=%f\n", time_step);
        printf("Num de Reynolds=%f\n", 1.0 / visc);
        printf("Execução em %d passos de tempo\n", time_max_number);

        // Incluir aqui mais um loop pro tempo (etapa 3)
        for (timestep_number = 1; condicao_de_parada(false); timestep_number++) {
            time += time_step;
            num_it = 0;

            // if (timestep_number == 10) {
            //     time_step = 0.1;
            // }
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

            printf("Problema nao linear com %d células, resolvido no passo de tempo %d, com %d iteracoes\n",
                   triangulation.n_active_cells(),
                   timestep_number,
                   num_it);

            // Salva solução no passo de tempo
            output_results(i);

            // Refinamento de malha
            if (timestep_number <= 3 || timestep_number % 10 == 0) {
                refine_grid();
            } else {
                printf("Pula Refinamento\n");
            }

            // Armazena solução em u^{n}
            prev_solution = solution;
            // Inicia solução anterior do método de Picard (u^{n+1 (bar)} = u^{n})
            old_solution = solution;
        }

        // Escreve como svg a malha final
        std::ofstream mesh_svg_file("mesh_final" + std::to_string(i) + ".svg");
        GridOut().write_svg(triangulation, mesh_svg_file);
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
    double time_step;
    double time;
    unsigned int timestep_number;
    double ref_error;
    double ref_error_prev;

    /**
     * Condição de parada do for do tempo
     * @is_aprox_estacionario True considera um erro entre u^n+1 e u^n menor que uma tolerancia,
     * False usa a contagem de passos até o tempo máximo indicado na variavel 'time_end'
     */
    bool condicao_de_parada(bool is_aprox_estacionario) {
        if (is_aprox_estacionario) {
            double delta = erro_norma_L2(solution, prev_solution);
            return timestep_number <= 2 || delta > 1e-6;
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

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;
            // right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
            //                                   rhs_values);

            fe_values[velocities].get_function_values(old_solution, old_solution_values);
            fe_values[velocities].get_function_values(prev_solution, prev_solution_values);

            // tau
            double h_mesh_min = GridTools::minimal_cell_diameter(triangulation);
            const double u_norm = old_solution.l2_norm();
            const double h_diff = h_mesh_min;
            const double h_conv = h_mesh_min;

            const double tau_u = pow(pow(1.0 / time_step, 2) + pow((2.0 * u_norm) / h_conv, 2) + ((4.0 * visc) / (3.0 * h_diff)), -0.5);

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

    void calcula_refinamento(Vector<float> estimated_error_per_cell) {
        double fracao_dividir = 0.3;
        double fracao_mesclar = 0.03 * ((timestep_number / 10.0) + 1.0);
        unsigned int limite_celulas = 16384;
        double min_area_cell = pow((omega_fim - omega_init) / 128.0, 2);
        int teste = 0;

        // Estratégia pronta do dealii
        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                        estimated_error_per_cell,
                                                        fracao_dividir,   // Dividir células com maior erro
                                                        fracao_mesclar,   // Mesclar células com menor erro
                                                        limite_celulas);  // Número máximo de células na malha

        // Desmarca refinamento de células com area menor que o limite
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int cell_no = cell->active_cell_index();
 
            if(cell->measure() <= min_area_cell){
                cell->clear_refine_flag();
                teste++;
            }
        }

        printf("%d celulas já chegaram no tamanho mínimo\n", teste);
    }

    //

    void refine_grid() {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        const FEValuesExtractors::Vector velocity(0);

        KellyErrorEstimator<dim>::estimate(
            dof_handler,
            QGauss<dim - 1>(degree + 1),
            std::map<types::boundary_id, const Function<dim> *>(),
            solution,
            estimated_error_per_cell,
            fe.component_mask(velocity));

        double max = estimated_error_per_cell[0];
        double min = estimated_error_per_cell[0];
        ref_error = 0;
        for (int i = 0; i < estimated_error_per_cell.size(); i++) {
            if (estimated_error_per_cell[i] >= max) {
                max = estimated_error_per_cell[i];
            } else if (estimated_error_per_cell[i] < min) {
                min = estimated_error_per_cell[i];
            }
            ref_error += estimated_error_per_cell[i];
        }
        // Calcula erro médio
        // ref_error /= estimated_error_per_cell.size();

        ref_error = max;

        printf("Max error: %f, Min error: %f\n", max, min);
        printf("*** Aplica Refinamento ***\n");

        calcula_refinamento(estimated_error_per_cell);

        // double fracao_dividir = 0.3;
        // double fracao_mesclar = 0.03 * ((timestep_number / 10.0) + 1.0);
        // unsigned int limite_celulas = 1000;

        // if (abs(ref_error - ref_error_prev) > 3e-3) {
        // if (timestep_number <= 3 || timestep_number % 10 == 0) {
        //printf("*** Aplica Refinamento div=%f, mesc=%f***\n", fracao_dividir, fracao_mesclar);
        // printf("*** Aplica Refinamento E: %f, Ep: %f, D: %f ***\n", ref_error, ref_error_prev, abs(ref_error - ref_error_prev));
        //  Faz o refinamento se o erro for alto entre um passo e outro
        // GridRefinement::refine_and_coarsen_fixed_number(triangulation,
        //                                                 estimated_error_per_cell,
        //                                                 fracao_dividir,   // Dividir células com maior erro
        //                                                 fracao_mesclar,   // Mesclar células com menor erro
        //                                                 limite_celulas);  // Número máximo de células na malha
        // ref_error_prev = ref_error;
        // } else {
        //     printf("*** Pula Refinamento ***\n");
        //     // printf("*** Pula Refinamento E: %f, Ep: %f, D: %f ***\n", ref_error, ref_error_prev, abs(ref_error - ref_error_prev));
        //     //  Se o erro for baixo, não faz refinamento
        // }

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

    void output_results(int i) const {
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

        std::string dir = "solution" + std::to_string(i);

        if (create_dir(dir)) {
            std::ofstream output(dir + "/solution_t" + std::to_string(timestep_number) + ".vtu");
            data_out.write_vtu(output);
        } else {
            printf("Erro ao criar diretorio\n");
        }
    }
};
}  // namespace CGNS
int main() {
    using namespace dealii;
    using namespace CGNS;

    const int dim = 2;
    const int i = 3;

    NavierStokesCG<dim> problem(1);
    problem.run(i);

    return 0;
}

/*
Função laplaciano

// No assemble
std::vector<Tensor<1,dim>> lap_phi_j_u(dofs_per_cell);
std::vector<Tensor<3,dim>> hess_phi_j_u(dofs_per_cell);

for (unsigned int i = 0; i < dofs_per_cell; ++i){
    hess_phi_j_u[i] = fe_values[velocities].hessian(i, q);
    for (int d = 0; d < dim; ++d)
        lap_phi_j_u[i][d] = trace(hess_phi_j_u[i][d]);
}

1º Fazer o SUPG nesse código transiente
2º PSPG
*/