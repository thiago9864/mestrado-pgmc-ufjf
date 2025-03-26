
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
template <int dim>
class RightHandSide : public Function<dim> {
   public:
    RightHandSide() : Function<dim>(dim + 1) {}

    double visc = 1.0;
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const {
        for (unsigned int i = 0; i < dim; ++i) {
            values[i] = dim * M_PI * M_PI * (visc - 1.0);
            if (i == dim - 1) {
                values[i] = -dim * M_PI * M_PI * ((dim - 1.0) * visc + 1.0);
            }

            for (unsigned int j = 0; j < dim; ++j) {
                if (i == j) {
                    values[i] *= sin(M_PI * p[j]);
                } else {
                    values[i] *= cos(M_PI * p[j]);
                }
            }
        }
        values[dim] = 0.0;
    }
};

template <int dim>
class ExactSolution : public Function<dim> {
   public:
    ExactSolution() : Function<dim>(dim + 1) {}
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const {
        Assert(values.size() == dim + 1,
               ExcDimensionMismatch(values.size(), dim + 1));

        for (unsigned int i = 0; i < dim; ++i) {
            values[i] = 1.0;
            if (i == dim - 1) {
                values[i] = -(dim - 1);
            }

            for (unsigned int j = 0; j < dim; ++j) {
                if (i == j) {
                    values[i] *= sin(M_PI * p[j]);
                } else {
                    values[i] *= cos(M_PI * p[j]);
                }
            }
        }

        values[dim] = dim * M_PI;
        for (int i = 0; i < dim; ++i) {
            values[dim] *= cos(M_PI * p[i]);
        }
    }
};

template <int dim>
class NavierStokesCG {
   public:
    NavierStokesCG(const unsigned int degree)
        : degree(degree), fe(FE_Q<dim>(degree), dim, FE_Q<dim>(degree - 1), 1), dof_handler(triangulation) {}
    void run(int i, ConvergenceTable &convergence_table) {
        make_grid_and_dofs(i);
        assemble_system();
        solve();
        compute_errors(convergence_table);
        output_results();
    }

   private:
    const unsigned int degree;
    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double> solution;
    BlockVector<double> system_rhs;
    AffineConstraints<double> constraints;

    void make_grid_and_dofs(int i) {
        GridGenerator::hyper_cube(triangulation, 0, 1);
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

        //        for (const auto &cell : dof_handler.active_cell_iterators())
        //            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        //                if (cell->face(f)->at_boundary()){
        //                    cell->face(f)->set_all_boundary_ids(0);
        //                }

        BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
        DoFTools::make_sparsity_pattern(dof_handler, dsp);

        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        constraints.clear();
        //        DoFTools::make_hanging_node_constraints (dof_handler, constraints);
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

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                              rhs_values);
            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const Tensor<2, dim> grad_phi_i_u = fe_values[velocities].gradient(i, q);
                    const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
                    const double div_phi_i_u = fe_values[velocities].divergence(i, q);
                    const double phi_i_p = fe_values[pressure].value(i, q);
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const Tensor<2, dim> grad_phi_j_u = fe_values[velocities].gradient(j, q);
                        const Tensor<1, dim> phi_j_u = fe_values[velocities].value(j, q);
                        const double div_phi_j_u = fe_values[velocities].divergence(j, q);
                        const double phi_j_p = fe_values[pressure].value(j, q);
                        local_matrix(i, j) += (scalar_product(grad_phi_i_u, grad_phi_j_u)  // (gradu,gradu)
                                               - phi_i_p * div_phi_j_u                     // -(q,divu)
                                               - div_phi_i_u * phi_j_p                     // -(p,divv)
                                               - 1e-8 * phi_i_p * phi_j_p)                 // -(p,q)   -> Isso elimina os valores zero na diagonal principal
                                              * fe_values.JxW(q);
                    }
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;

                    local_rhs(i) += fe_values.shape_value(i, q) * rhs_values[q](component_i) * fe_values.JxW(q);  // (f, v)
                }
            }
            cell->get_dof_indices(local_dof_indices);
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
        //                PreconditionJacobi <SparseMatrix<double>> precondition;
        //                precondition.initialize (system_matrix, 0.6);
        //
        //                SolverControl solver_control (system_matrix.m()*10, 1e-10*system_rhs.l2_norm());
        //                SolverCG <> solver(solver_control);
        //                solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity() );
        //
        //                std::cout << std::endl << "   Number of BiCGStab iterations: " << solver_control.last_step()
        //                          << std::endl
        //                          << std::endl;

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
        std::ofstream output("solution.vtu");
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
