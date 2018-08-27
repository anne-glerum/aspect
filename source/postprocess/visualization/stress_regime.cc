/*
  Copyright (C) 2016 - 2017 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/postprocess/visualization/stress_regime.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/utilities.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      void
      StressRegime<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double> > &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        // Output is a dim+1 vector of where the first three entries represent the maximum horizontal
        // compressive stress and the fourth has a value of 1, 2 or 3, where 1 is normal faulting, 2 strike-slip and 3 thrust faulting
        Assert ((computed_quantities[0].size() == dim+1),
                ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,   ExcInternalError());
        Assert (input_data.solution_gradients[0].size() == this->introspection().n_components,  ExcInternalError());

        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection());
        MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
                                                     this->n_compositional_fields());

        // Compute the viscosity...
        this->get_material_model().evaluate(in, out);

        // ...and use it to compute the stresses and from that the
        // maximum compressive stress direction
        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            const SymmetricTensor<2,dim> strain_rate = in.strain_rate[q];
            const SymmetricTensor<2,dim> compressible_strain_rate
              = (this->get_material_model().is_compressible()
                 ?
                 strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                 :
                 strain_rate);

            const double eta = out.viscosities[q];

            // first compute the stress tensor, ignoring the pressure
            // for the moment (the pressure has no effect on the
            // direction since it just adds a multiple of the identity
            // matrix to the stress, but because it is large, it may
            // lead to numerical instabilities)
            //
            // note that the *compressive* stress is simply the
            // negative stress
            const SymmetricTensor<2,dim> compressive_stress = -2*eta*compressible_strain_rate;

            // then find a set of (dim-1) horizontal, unit-length, mutually orthogonal vectors
            const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector (in.position[q]);
            const Tensor<1,dim> vertical_direction = gravity/gravity.norm();
            std::array<Tensor<1,dim>,dim-1 > orthogonal_directions
              = Utilities::orthogonal_vectors(vertical_direction);
            for (unsigned int i=0; i<orthogonal_directions.size(); ++i)
              orthogonal_directions[i] /= orthogonal_directions[i].norm();

            double stress_regime = std::numeric_limits<double>::quiet_NaN();
            Tensor<1,dim> maximum_horizontal_compressive_stress;
            switch (dim)
              {
                // in 2d, there is only one horizontal direction, and
                // we have already computed it above. give it the
                // length of the compressive_stress (now taking into account the
                // pressure) in this direction
                case 2:
                {
                  const double maximum_horizontal_compressive_stress_magnitude = orthogonal_directions[0] *
                                                           ((compressive_stress
                                                             -
                                                             in.pressure[q] * unit_symmetric_tensor<dim>()) * orthogonal_directions[0]);

                  const double vertical_compressive_stress_magnitude = vertical_direction *
                                                           ((compressive_stress
                                                             -
                                                             in.pressure[q] * unit_symmetric_tensor<dim>()) *
                                                               vertical_direction);

                  // normal faulting
                  if (vertical_compressive_stress_magnitude > maximum_horizontal_compressive_stress_magnitude)
                    stress_regime = 1.;
                  // thrust faulting
                  else
                    stress_regime = 3.;

                  maximum_horizontal_compressive_stress = orthogonal_directions[0] * maximum_horizontal_compressive_stress_magnitude;

                  break;
                }

                // in 3d, use the formulas discussed in the
                // documentation of the plugin below
                case 3:
                {
                  const double a = orthogonal_directions[0] *
                                   (compressive_stress *
                                    orthogonal_directions[0]);
                  const double b = orthogonal_directions[1] *
                                   (compressive_stress *
                                    orthogonal_directions[1]);
                  const double c = orthogonal_directions[0] *
                                   (compressive_stress *
                                    orthogonal_directions[1]);

                  // compute the two stationary points of f(alpha)
                  const double alpha_1 = 1./2 * std::atan2 (c, a-b);
                  const double alpha_2 = alpha_1 + numbers::PI/2;

                  // then check the sign of f''(alpha) to determine
                  // which of the two stationary points is the maximum
                  const double f_double_prime_1 = 2*(b-a)*std::cos(2*alpha_1)
                                                  - 2*c*sin(2*alpha_1);
                  double alpha;
                  if (f_double_prime_1 < 0)
                    alpha = alpha_1;
                  else
                    {
                      Assert (/* f_double_prime_2 = */
                        2*(b-a)*std::cos(2*alpha_2) - 2*c*sin(2*alpha_2) <= 0,
                        ExcInternalError());
                      alpha = alpha_2;
                    }

                  // then re-assemble the maximum horizontal compressive_stress
                  // direction from alpha and the two horizontal
                  // directions
                  const Tensor<1,dim> n = std::cos(alpha) * orthogonal_directions[0] +
                                          std::sin(alpha) * orthogonal_directions[1];

                  // finally compute the actual direction * magnitude,
                  // now taking into account the pressure (with the
                  // correct sign in front of the pressure for the
                  // *compressive* stress)
                  //
                  // the magnitude is computed as discussed in the
                  // description of the plugin below
                  const double maximum_horizontal_compressive_stress_magnitude
                    = (n * ((compressive_stress
                             -
                             in.pressure[q] * unit_symmetric_tensor<dim>()) * n));

                  const Tensor<1,dim> n_perp = std::sin(alpha) * orthogonal_directions[0] -
                                               std::cos(alpha) * orthogonal_directions[1];

                  const double minimum_horizontal_compressive_stress_magnitude
                    = (n_perp * ((compressive_stress
                                  -
                                  in.pressure[q] * unit_symmetric_tensor<dim>()) * n_perp));

                  const double vertical_compressive_stress_magnitude
                    = (vertical_direction * ((compressive_stress
                        -
                        in.pressure[q] * unit_symmetric_tensor<dim>()) * vertical_direction));

                    // normal faulting
                    if (vertical_compressive_stress_magnitude > maximum_horizontal_compressive_stress_magnitude &&
                        maximum_horizontal_compressive_stress_magnitude > minimum_horizontal_compressive_stress_magnitude)
                      stress_regime = 1.;
                    // strike-slip faulting
                    else if (maximum_horizontal_compressive_stress_magnitude > vertical_compressive_stress_magnitude &&
                        vertical_compressive_stress_magnitude > minimum_horizontal_compressive_stress_magnitude)
                      stress_regime = 2.;
                    // thrust faulting
                    else if (maximum_horizontal_compressive_stress_magnitude > minimum_horizontal_compressive_stress_magnitude &&
                        minimum_horizontal_compressive_stress_magnitude > vertical_compressive_stress_magnitude)
                      stress_regime = 3.;

                  maximum_horizontal_compressive_stress = n * (maximum_horizontal_compressive_stress_magnitude - minimum_horizontal_compressive_stress_magnitude);


                  break;
                }


                default:
                  Assert (false, ExcNotImplemented());
              }

            for (unsigned int i=0; i<dim; ++i)
              computed_quantities[q](i) = maximum_horizontal_compressive_stress[i];
            computed_quantities[q](dim) = stress_regime;
          }
      }


      template <int dim>
      std::vector<std::string>
      StressRegime<dim>::get_names () const
      {
        std::vector<std::string> names(dim+1);
        names[0] = "sigma_H_x";
        names[dim-1] = "sigma_H_z";
        names[1] = "sigma_H_y";
        names[dim] = "stress_regime";
        return names;
      }


      template <int dim>
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      StressRegime<dim>::get_data_component_interpretation () const
      {
          std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation
            (dim+1,DataComponentInterpretation::component_is_part_of_vector);
          interpretation[dim] = DataComponentInterpretation::component_is_scalar;
        return interpretation;
      }



      template <int dim>
      UpdateFlags
      StressRegime<dim>::get_needed_update_flags () const
      {
        return update_gradients | update_values | update_q_points;
      }

    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(StressRegime,
                                                  "stress regime",
                                                  "A plugin that computes the direction of the "
                                                  "maximum horizontal component of the compressive stress as a vector "
                                                  "field, scaled with a value that indicates the principle mode of deformation. "
                                                  "A value of 1 indicates normal faulting, 2 strike-slip and 3 thrust faulting. "
                                                  "Recall that the "
                                                  "\\textit{compressive} stress is simply the negative stress, "
                                                  "$\\sigma_c=-\\sigma=-\\left["
                                                  "     2\\eta (\\varepsilon(\\mathbf u)"
                                                  "             - \\frac 13 (\\nabla \\cdot \\mathbf u) I)"
                                                  "     + pI\\right]$."
                                                  "\n\n"
                                                  "Following \\cite{LundTownend07}, we define the maximum horizontal "
                                                  "stress direction as that \\textit{horizontal} direction "
                                                  "$\\mathbf n$ that maximizes $\\mathbf n^T \\sigma_c \\mathbf n$. We "
                                                  "call a vector \\textit{horizontal} if it is perpendicular to the "
                                                  "gravity vector $\\mathbf g$."
                                                  "\n\n"
                                                  "In two space dimensions, $\\mathbf n$ is simply a vector that "
                                                  "is horizontal (we choose one of the two possible choices). "
                                                  "This direction is then scaled by the size of the horizontal stress "
                                                  "in this direction, i.e., the plugin outputs the vector "
                                                  "$\\mathbf w = (\\mathbf n^T \\sigma_c \\mathbf n) \\; \\mathbf n$."
                                                  "\n\n"
                                                  "In three space dimensions, given two horizontal, perpendicular, "
                                                  "unit length, but otherwise arbitrarily chosen vectors "
                                                  "$\\mathbf u,\\mathbf v$, we can express "
                                                  "$\\mathbf n = (\\cos \\alpha)\\mathbf u + (\\sin\\alpha)\\mathbf v$ "
                                                  "where $\\alpha$ maximizes the expression "
                                                  "\\begin{align*}"
                                                  "  f(\\alpha) = \\mathbf n^T \\sigma_c \\mathbf n"
                                                  "  = (\\mathbf u^T \\sigma_c \\mathbf u)(\\cos\\alpha)^2"
                                                  "    +2(\\mathbf u^T \\sigma_c \\mathbf v)(\\cos\\alpha)(\\sin\\alpha)"
                                                  "    +(\\mathbf v^T \\sigma_c \\mathbf v)(\\sin\\alpha)^2."
                                                  "\\end{align*}"
                                                  "\n\n"
                                                  "The maximum of $f(\\alpha)$ is attained where $f'(\\alpha)=0$. "
                                                  "Evaluating the derivative and using trigonometric identities, "
                                                  "one finds that $\\alpha$ has to satisfy the equation "
                                                  "\\begin{align*}"
                                                  "  \\tan(2\\alpha) = \\frac{\\mathbf u^T \\sigma_c \\mathbf v}"
                                                  "                          {\\mathbf u^T \\sigma_c \\mathbf u "
                                                  "                           - \\mathbf v^T \\sigma_c \\mathbf v}."
                                                  "\\end{align*}"
                                                  "Since the transform $\\alpha\\mapsto\\alpha+\\pi$ flips the "
                                                  "direction of $\\mathbf n$, we only need to seek a solution "
                                                  "to this equation in the interval $\\alpha\\in[0,\\pi)$. "
                                                  "These are given by "
                                                  "$\\alpha_1=\\frac 12 \\arctan \\frac{\\mathbf u^T \\sigma_c "
                                                  "\\mathbf v}{\\mathbf u^T \\sigma_c \\mathbf u - "
                                                  "\\mathbf v^T \\sigma_c \\mathbf v}$ and "
                                                  "$\\alpha_2=\\alpha_1+\\frac{\\pi}{2}$, one of which will "
                                                  "correspond to a minimum and the other to a maximum of "
                                                  "$f(\\alpha)$. One checks the sign of "
                                                  "$f''(\\alpha)=-2(\\mathbf u^T \\sigma_c \\mathbf u - "
                                                  "\\mathbf v^T \\sigma_c \\mathbf v)\\cos(2\\alpha) "
                                                  "- 2 (\\mathbf u^T \\sigma_c \\mathbf v) \\sin(2\\alpha)$ for "
                                                  "each of these to determine the $\\alpha$ that maximizes "
                                                  "$f(\\alpha)$, and from this immediately arrives at the correct "
                                                  "form for the maximum horizontal stress $\\mathbf n$."
                                                  "\n\n"
                                                  "The stress regime is determined based on the magnitudes of the "
                                                  "maximum ($\\sigma_H$) and minimum ($\\sigma_h$) horizontal stress "
                                                  "and the vertical stress, "
                                                  "where $\\sigma_v>\\sigma_H>\\sigma_h$ defines normal faulting, "
                                                  "$\\sigma_H>\\sigma_v>\\sigma_h$ strike-slip faulting and "
                                                  "$\\sigma_H>\\sigma_h>\\sigma_v$ thrust faulting. "
                                                 )
    }
  }
}
