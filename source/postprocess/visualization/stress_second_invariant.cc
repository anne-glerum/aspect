/*
  Copyright (C) 2011 - 2022 by the authors of the ASPECT code.

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



#include <aspect/postprocess/visualization/stress_second_invariant.h>

#include <aspect/material_model/rheology/elasticity.h>
#include <aspect/material_model/visco_plastic.h>
#include <aspect/material_model/viscoelastic.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      StressSecondInvariant<dim>::
      StressSecondInvariant ()
        :
        DataPostprocessorScalar<dim> ("stress_second_invariant",
                                      update_values | update_gradients | update_quadrature_points),
        Interface<dim>("Pa")
      {}



      template <int dim>
      void
      StressSecondInvariant<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert(computed_quantities.size() == n_quadrature_points, ExcInternalError());
        Assert(computed_quantities[0].size() == 1, ExcInternalError());
        Assert(input_data.solution_values[0].size() == this->introspection().n_components, ExcInternalError());
        Assert(input_data.solution_gradients[0].size() == this->introspection().n_components, ExcInternalError());

        // Create the material model inputs and outputs to
        // retrieve the current viscosity.
        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection(),
                                                   /*compute_strain_rate = */ true);

        in.requested_properties = MaterialModel::MaterialProperties::viscosity;

        MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
                                                     this->n_compositional_fields());

        this->get_material_model().evaluate(in, out);

        for (unsigned int q = 0; q < n_quadrature_points; ++q)
          {
            const SymmetricTensor<2, dim> strain_rate = in.strain_rate[q];
            const SymmetricTensor<2, dim> deviatoric_strain_rate = (this->get_material_model().is_compressible()
                                                                    ? strain_rate - 1. / 3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                                                                    : strain_rate);

            const double eta = out.viscosities[q];

            // Compressive stress is positive in geoscience applications.
            SymmetricTensor<2, dim> stress = in.pressure[q] * unit_symmetric_tensor<dim>();

            if (this->get_parameters().enable_elasticity == true)
              {
                // Visco-elastic stresses are stored on the fields
                SymmetricTensor<2, dim> stress_0;
                stress_0[0][0] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xx")];
                stress_0[1][1] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_yy")];
                stress_0[0][1] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xy")];

                if (dim == 3)
                  {
                    stress_0[2][2] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_zz")];
                    stress_0[0][2] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xz")];
                    stress_0[1][2] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_yz")];
                  }

                const MaterialModel::ElasticAdditionalOutputs<dim> *elastic_out = out.template get_additional_output<MaterialModel::ElasticAdditionalOutputs<dim>>();

                const double shear_modulus = elastic_out->elastic_shear_moduli[q];

                // $\eta_{el} = G \Delta t_{el}$
                double elastic_viscosity = this->get_timestep() * shear_modulus;
                if (Plugins::plugin_type_matches<MaterialModel::ViscoPlastic<dim>>(this->get_material_model()))
                  {
                    const MaterialModel::ViscoPlastic<dim> &vp = Plugins::get_plugin_as_type<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model());
                    elastic_viscosity = vp.get_elastic_viscosity(shear_modulus);
                  }

                // Apply the stress update to get the total stress of timestep t.
                stress = 2. * eta * (deviatoric_strain_rate + stress_0 / (2. * elastic_viscosity));
              }
            else
              {
                stress += -2. * eta * deviatoric_strain_rate;
              }

            // Compute the deviatoric stress tensor after elastic stresses were added.
            const SymmetricTensor<2, dim> deviatoric_stress = deviator(stress);

            // Compute the second moment invariant of the deviatoric stress
            // in the same way as the second moment invariant of the deviatoric
            // strain rate is computed in the viscoplastic material model.
            // TODO check that this is valid for the compressible case.
            const double stress_invariant = std::sqrt(std::max(-second_invariant(deviatoric_stress), 0.));

            computed_quantities[q](0) = stress_invariant;
          }

        // average the values if requested
        const auto &viz = this->get_postprocess_manager().template get_matching_postprocessor<Postprocess::Visualization<dim>>();
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
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(StressSecondInvariant,
                                                  "stress second invariant",
                                                  "A visualization output object that outputs "
                                                  "the second moment invariant of the deviatoric stress tensor."
                                                  "\n\n"
                                                  "Physical units: \\si{\\pascal}.")
    }
  }
}
