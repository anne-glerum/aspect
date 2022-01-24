/*
  Copyright (C) 2016 - 2019 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/strain_rate_point_element.h>
#include <aspect/adiabatic_conditions/interface.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      StrainRatePE<dim>::
      StrainRatePE ()
        :
        DataPostprocessor<dim> ()
      {}

      template <int dim>
      std::vector<std::string>
      StrainRatePE<dim>::
      get_names () const
      {
        std::vector<std::string> solution_names;
        solution_names.emplace_back("point_wise_strain_rate");
        solution_names.emplace_back("element_wise_strain_rate");
        return solution_names;
      }


      template <int dim>
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      StrainRatePE<dim>::
      get_data_component_interpretation () const
      {
        std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(2,
            DataComponentInterpretation::component_is_scalar);

        return interpretation;
      }


      template <int dim>
      UpdateFlags
      StrainRatePE<dim>::
      get_needed_update_flags () const
      {
        return update_gradients | update_quadrature_points;
      }


      template <int dim>
      void
      StrainRatePE<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        Assert (computed_quantities[0].size() == 2,                   ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,           ExcInternalError());
        Assert (input_data.solution_gradients[0].size() == this->introspection().n_components,          ExcInternalError());

        // to hold the point-wise values to be averaged afterwards
        std::vector<Vector<double>> tmp_computed_quantities(n_quadrature_points, Vector<double>(1));

        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            Tensor<2,dim> grad_u;
            for (unsigned int d=0; d<dim; ++d)
              grad_u[d] = input_data.solution_gradients[q][d];

            const SymmetricTensor<2,dim> strain_rate = symmetrize (grad_u);
            // point-wise
            computed_quantities[q](0) = std::sqrt(std::fabs(second_invariant(deviator(strain_rate))));
            // element-wise values to be averaged below
            tmp_computed_quantities[q](0) = std::sqrt(std::fabs(second_invariant(deviator(strain_rate))));
          }

        // average the values if requested
        average_quantities(tmp_computed_quantities);

        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            computed_quantities[q](1) = tmp_computed_quantities[q](0);
          }


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
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(StrainRatePE,
                                                  "strain rate point and element wise",
                                                  "A visualization output "
                                                  "object that generates "
                                                  "the second invariant of the strain rate "
                                                  "on the nodes as well as an element-average.")
    }
  }
}
