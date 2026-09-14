/*
  Copyright (C) 2011 - 2025 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/rotation_rate_tensor.h>
#include <aspect/material_model/rheology/elasticity.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      RotationRateTensor<dim>::
      RotationRateTensor ()
        :
        DataPostprocessorTensor<dim> ("rotation_rate_tensor",
                                      update_values | update_gradients | update_quadrature_points),
        Interface<dim>("1/s")
      {}



      template <int dim>
      void
      RotationRateTensor<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        AssertThrow(this->get_parameters().enable_elasticity == true,
                    ExcMessage("Visualization plugin ``rotation rate tensor'' only works if 'Enable elasticity' is set to true."));
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        Assert ((computed_quantities[0].size() == Tensor<2,dim>::n_independent_components),
                ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,   ExcInternalError());
        Assert (input_data.solution_gradients[0].size() == this->introspection().n_components,  ExcInternalError());

        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection());
        MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
                                                     this->n_compositional_fields());

        // We do not need to compute anything but the viscosity and the additional outputs
        in.requested_properties = MaterialModel::MaterialProperties::viscosity | MaterialModel::MaterialProperties::additional_outputs;

        this->get_material_model().create_additional_named_outputs(out);

        // Compute the additional outputs
        this->get_material_model().evaluate(in, out);

        // ...and use them to compute the stresses
        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            // Get the total deviatoric stress from the material model.
            const std::shared_ptr<const MaterialModel::ElasticAdditionalOutputs<dim>> elastic_additional_out
              = out.template get_additional_output_object<MaterialModel::ElasticAdditionalOutputs<dim>>();

            Assert(elastic_additional_out != nullptr, ExcMessage("Elastic Additional Outputs are needed for the 'rotation tensor' postprocessor, but they have not been created."));

            const Tensor<2,dim> rotation = elastic_additional_out->rotation_tensor[q];

            for (unsigned int d=0; d<dim; ++d)
              for (unsigned int e=0; e<dim; ++e)
                computed_quantities[q][Tensor<2,dim>::component_to_unrolled_index(TableIndices<2>(d,e))]
                  = rotation[d][e];
          }

        // average the values if requested
        const auto &viz = this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::Visualization<dim>>();
        if (!viz.output_pointwise_stress_and_strain())
          average_quantities(computed_quantities);
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
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(RotationRateTensor,
                                                  "rotation rate tensor",
                                                  "A visualization output object that generates output "
                                                  "for the 4 (in 2d) or 9 (in 3d) components of the rotation "
                                                  "tensor."
                                                  "\n\n"
                                                  "This postprocessor outputs the quantity computed herein as "
                                                  "a tensor, i.e., programs such as VisIt or Pararview can "
                                                  "visualize it as tensors represented by ellipses, not just "
                                                  "as individual fields. That said, you can also visualize "
                                                  "individual tensor components, by noting that the "
                                                  "components that are written to the output file correspond to "
                                                  "the tensor components $t_{xx}, t_{xy}, t_{yx}, t_{yy}$ (in 2d) "
                                                  "or  $t_{xx}, t_{xy}, t_{xz}, t_{yx}, t_{yy}, t_{yz}, t_{zx}, t_{zy}, "
                                                  "t_{zz}$ (in 3d) of a tensor $t$ in a Cartesian coordinate system. "
                                                  "Even though the tensor we output is symmetric, the output contains "
                                                  "all components of the tensor because that is what the file format "
                                                  "requires."
                                                  "\n\n"
                                                  "Physical units: $\\text{1/s}$.")
    }
  }
}
