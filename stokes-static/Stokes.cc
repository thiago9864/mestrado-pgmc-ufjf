
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

#include <fstream>
#include <iostream>
namespace CGNS {
using namespace dealii;

// Parâmetros do problema
double visc = 0.025;

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

/*
 * Classe que contem a representação da solução exata do problema
 */
template <int dim>
class ExactSolution : public Function<dim> {
   public:
    ExactSolution() : Function<dim>(dim + 1) {}

    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const {
        Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));

        double re = 1.0 / visc;
        double delta = (re / 2.0) - sqrt((re * re / 4.0) + (4.0 * M_PI * M_PI));
        double p0 = (-exp(-delta) + exp(3.0 * delta)) / (8.0 * delta);

        // ux
        values[0] = 1.0 - exp(delta * p[0]) * cos(2.0 * M_PI * p[1]);

        // uy
        values[1] = (delta / (2.0 * M_PI)) * exp(delta * p[0]) * sin(2.0 * M_PI * p[1]);

        // p
        values[2] = -0.5 * exp(2.0 * delta * p[0]) + p0;
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
          fe(FE_Q<dim>(degree), dim, FE_Q<dim>(degree - 1), 1),
          dof_handler(triangulation) {}

    /*
     * Função de execução principal
     * @i Grau de refinamento da malha -> 2^i elementos
     * @convergence_table Tabela de convergência
     */
    void run(int i, ConvergenceTable &convergence_table) {
        int num_it = 0;
        double delta;
        // Inicializa malha
        make_grid_and_dofs(i);

        // Incluir aqui mais um loop pro tempo (etapa 3)

        // Metodo de Picard
        do {
            system_matrix = 0;
            system_rhs = 0;

            assemble_system();
            solve();

            // Calcula diferença entre as soluções com a norma L2 discreta
            // delta = (solution.block(0) - old_solution.block(0)).l2_norm();
            delta = erro_norma_L2();

            // Salva cópia da solução obtida
            old_solution = solution;

            num_it++;
            if (num_it > 100) {
                printf("Número de iteracoes acima do limite de 100\n");
                printf("Delta alcançado: %f\n", delta);
                exit(3);
            }
        } while (delta > 1e-6);

        printf("Problema nao linear resolvido com %d iteracoes\n", num_it);

        compute_errors(convergence_table);
        output_results(i);
    }

   private:
    const unsigned int degree;
    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double> solution;
    BlockVector<double> old_solution;
    BlockVector<double> system_rhs;
    AffineConstraints<double> constraints;

    /*
     * Calcula erro na norma L2 entre duas soluções
     */
    double erro_norma_L2() {
        double erroL2 = 0.0;
        double diff;

        QGauss<dim> quadrature_formula(degree + 1);
        QGauss<dim - 1> face_quadrature_formula(degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

        const unsigned int n_q_points = quadrature_formula.size();

        // posições na celula atual
        std::vector<Tensor<1, dim>> old_solution_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_values(n_q_points);

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);

            fe_values[velocities].get_function_values(solution, solution_values);
            fe_values[velocities].get_function_values(old_solution, old_solution_values);

            diff = 0;
            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int d = 0; d < dim; ++d) {
                    diff += (pow(old_solution_values[q][d] - solution_values[q][d], 2) * fe_values.JxW(q));
                }
            }
            erroL2 += diff;
        }

        return sqrt(erroL2);
    }

    /**
     * Inicializa a malha e inclui os pontos locais e globais de cada nó
     */
    void make_grid_and_dofs(int i) {
        // Aqui define o tamanho do domínio, nessa função hyper_cube
        GridGenerator::hyper_cube(triangulation, -0.5, 1.5);
        triangulation.refine_global(i);
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

        solution.reinit(dofs_per_block);
        system_rhs.reinit(dofs_per_block);
        old_solution.reinit(dofs_per_block);

        BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
        DoFTools::make_sparsity_pattern(dof_handler, dsp);

        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        // tirar a condição de contorno daqui e criar uma função separada (etapa 3)
        constraints.clear();

        ExactSolution<dim> solution_function;
        FEValuesExtractors::Vector velocity(0);
        VectorTools::interpolate_boundary_values(dof_handler, 0, solution_function, constraints, fe.component_mask(velocity));
        constraints.close();
    }

    void assemble_system() {
        QGauss<dim> quadrature_formula(degree + 1);
        QGauss<dim - 1> face_quadrature_formula(degree + 1);
        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients |
                                    update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe,
                                         face_quadrature_formula,
                                         update_values | update_normal_vectors |
                                             update_quadrature_points |
                                             update_JxW_values);
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        const RightHandSide<dim> right_hand_side;

        std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

        // posições na celula atual
        std::vector<Tensor<1, dim>> old_solution_values(n_q_points);

        const FEValuesExtractors::Vector velocities(0);  //(x,y,?)
        const FEValuesExtractors::Scalar pressure(dim);  //(?,?,p)
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                              rhs_values);

            fe_values[velocities].get_function_values(old_solution, old_solution_values);

            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    // 1ª equação
                    const Tensor<2, dim> grad_phi_i_u = fe_values[velocities].gradient(i, q);  // grad_v
                    const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);          // v
                    const double div_phi_i_u = fe_values[velocities].divergence(i, q);         // div v

                    // 2ª equação
                    const double phi_i_p = fe_values[pressure].value(i, q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const Tensor<2, dim> grad_phi_j_u = fe_values[velocities].gradient(j, q);  // grad_u
                        // const Tensor<1, dim> phi_j_u = fe_values[velocities].value(j, q);
                        const double div_phi_j_u = fe_values[velocities].divergence(j, q);
                        const double phi_j_p = fe_values[pressure].value(j, q);

                        // O indice da matriz local determina quem, a respeito de i,j vai ser
                        // a solução ou função teste. A primeira componente do local_matrix
                        // é a função teste (i), e função solução (j)
                        local_matrix(i, j) += (visc * scalar_product(grad_phi_i_u, grad_phi_j_u)  // 2 * mu * (grad_v, grad_u)
                                               + grad_phi_j_u * old_solution_values[q] * phi_i_u  //  (gradu,u_old) * v // Termo convectivo
                                               - phi_i_p * div_phi_j_u                            // -(q,divu)
                                               - div_phi_i_u * phi_j_p                            // -(p,divv)
                                               - 1e-8 * phi_i_p * phi_j_p)                        // -(p,q)   -> Isso elimina os valores zero na diagonal principal
                                              * fe_values.JxW(q);                                 // Esse J e W já incluem a função peso e W da integração
                    }
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

    void compute_errors(ConvergenceTable &convergence_table) {
        const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                         dim + 1);
        ExactSolution<dim> exact_solution;
        Vector<double> cellwise_errors(triangulation.n_active_cells());
        QGauss<dim> quadrature(degree + 1);

        const double mean_pressure = VectorTools::compute_mean_value(dof_handler, quadrature, solution, dim);
        std::cout << "   Note: The mean value was adjusted by " << -mean_pressure << std::endl;
        solution.block(1).add(-mean_pressure);

        VectorTools::integrate_difference(dof_handler,
                                          solution,
                                          exact_solution,
                                          cellwise_errors,
                                          quadrature,
                                          VectorTools::L2_norm,
                                          &pressure_mask);
        const double p_l2_error =
            VectorTools::compute_global_error(triangulation,
                                              cellwise_errors,
                                              VectorTools::L2_norm);
        VectorTools::integrate_difference(dof_handler,
                                          solution,
                                          exact_solution,
                                          cellwise_errors,
                                          quadrature,
                                          VectorTools::L2_norm,
                                          &velocity_mask);
        const double u_l2_error =
            VectorTools::compute_global_error(triangulation,
                                              cellwise_errors,
                                              VectorTools::L2_norm);

        std::cout << "Errors: ||e_u||_L2 = " << u_l2_error
                  << ",   ||e_p||_L2 = " << p_l2_error
                  << std::endl
                  << std::endl;

        convergence_table.add_value("cells", triangulation.n_active_cells());
        convergence_table.add_value("L2_u", u_l2_error);
        convergence_table.add_value("L2_p", p_l2_error);
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
        std::ofstream output("solution" + std::to_string(i) + ".vtu");
        data_out.write_vtu(output);
    }
};
}  // namespace CGNS
int main() {
    using namespace dealii;
    using namespace CGNS;

    ConvergenceTable convergence_table;

    const int dim = 2;

    for (int i = 2; i < 6; ++i) {
        NavierStokesCG<dim> problem(2);
        problem.run(i, convergence_table);
    }
    convergence_table.set_precision("L2_u", 3);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.evaluate_convergence_rates("L2_u", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.set_precision("L2_p", 3);
    convergence_table.set_scientific("L2_p", true);
    convergence_table.evaluate_convergence_rates("L2_p", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.write_text(std::cout);

    std::ofstream data_output("taxas.dat");
    convergence_table.write_text(data_output);

    std::ofstream tex_output("taxas.tex");
    convergence_table.write_tex(tex_output);

    return 0;
}
