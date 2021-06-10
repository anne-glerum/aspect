/*
  Copyright (C) 2011 - 2020 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/surface_velocity.h>



namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      void
      SurfaceVelocity<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double> > &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        Assert ((computed_quantities[0].size() == dim),
                ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,   ExcInternalError());

            const double velocity_scaling_factor =
              this->convert_output_to_years() ? year_in_seconds : 1.0;

        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            for (unsigned int d=0; d<dim; ++d)
              computed_quantities[q](d) = input_data.solution_values[q][d] * velocity_scaling_factor;
          }

      }


      template <int dim>
      std::vector<std::string>
      SurfaceVelocity<dim>::get_names () const
      {
        std::vector<std::string> names (dim, "surface_velocity");
//        std::vector<std::string> names;
//              names.emplace_back("surface_velocity_x");
//              names.emplace_back("surface_velocity_y");
//
//            if (dim == 3)
//              names.emplace_back("surface_velocity_z");

        return names;
      }


      template <int dim>
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      SurfaceVelocity<dim>::get_data_component_interpretation () const
      {
        return
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
          (dim,
           DataComponentInterpretation::component_is_part_of_vector);
//           DataComponentInterpretation::component_is_scalar);
      }



      template <int dim>
      UpdateFlags
      SurfaceVelocity<dim>::get_needed_update_flags () const
      {
        return update_values | update_quadrature_points;
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
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(SurfaceVelocity,
                                                  "surface velocity",
                                                  "A visualization output object that generates output "
                                                  "on the surface of the domain "
                                                  "for the 2 (in 2d) or 3 (in 3d) components of the velocity "
                                                  "vector. ")
    }
  }
}
