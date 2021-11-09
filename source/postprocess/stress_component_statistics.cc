/*
  Copyright (C) 2011 - 2019 by the authors of the ASPECT code.

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

#include <aspect/postprocess/stress_component_statistics.h>
#include <aspect/material_model/rheology/elasticity.h>
#include <aspect/material_model/visco_plastic.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    StressComponentStatistics<dim>::execute (TableHandler &statistics)
    {
      // TODO: this postprocessor is no longer necessary and can be removed from the PR.
      if (this->n_compositional_fields() == 0)
        return std::pair<std::string,std::string>();

      // create a quadrature formula based on the compositional element alone.
      // be defensive about determining that a compositional field actually exists
      AssertThrow (this->introspection().base_elements.compositional_fields
                   != numbers::invalid_unsigned_int,
                   ExcMessage("This postprocessor cannot be used without compositional fields."));
      const QGauss<dim> quadrature_formula (this->get_fe().base_element(this->introspection().base_elements.compositional_fields).degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_gradients |
                               update_quadrature_points);

      std::vector<double> compositional_values(n_q_points);

      std::vector<double> local_min_stress_components(this->n_compositional_fields(),
                                                      std::numeric_limits<double>::max());
      std::vector<double> local_max_stress_components(this->n_compositional_fields(),
                                                      std::numeric_limits<double>::lowest());

      typename MaterialModel::Interface<dim>::MaterialModelInputs in(n_q_points,
                                                                     this->n_compositional_fields());
      typename MaterialModel::Interface<dim>::MaterialModelOutputs out(n_q_points,
                                                                       this->n_compositional_fields());

      // compute the integral quantities by quadrature
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);

            in.reinit(fe_values,
                      cell,
                      this->introspection(),
                      this->get_solution());

            in.requested_properties = MaterialModel::MaterialProperties::viscosity;

            this->get_material_model().fill_additional_material_model_inputs(in,
                                                                             this->get_solution(),
                                                                             fe_values,
                                                                             this->introspection());

            this->get_material_model().create_additional_named_outputs(out);

            this->get_material_model().evaluate(in, out);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const SymmetricTensor<2, dim> strain_rate = in.strain_rate[q];
                const SymmetricTensor<2, dim> deviatoric_strain_rate = (this->get_material_model().is_compressible()
                                                                        ? strain_rate - 1. / 3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                                                                        : strain_rate);

                const double eta = out.viscosities[q];

                // Compressive stress is positive in geoscience applications
                SymmetricTensor<2, dim> stress = -2. * eta * deviatoric_strain_rate +
                                                 in.pressure[q] * unit_symmetric_tensor<dim>();

                // Add elastic stresses if existent
                if (this->get_parameters().enable_elasticity == true)
                  {
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

                    // The total stress of timestep t.
                    stress = 2. * eta * (deviatoric_strain_rate + stress_0 / (2. * elastic_viscosity));
                  }

                // Compute the deviatoric stress
                stress = deviator(stress);

                for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
                  {
                    local_min_stress_components[c] = std::min<double>(local_min_stress_components[c], stress[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)]);
                    local_max_stress_components[c] = std::max<double>(local_max_stress_components[c], stress[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)]);
                  }
              }

          }

      // now do the reductions over all processors
      std::vector<double> global_min_stress_components (this->n_compositional_fields(),
                                                        std::numeric_limits<double>::max());
      std::vector<double> global_max_stress_components (this->n_compositional_fields(),
                                                        std::numeric_limits<double>::lowest());
      {
        Utilities::MPI::min (local_min_stress_components,
                             this->get_mpi_communicator(),
                             global_min_stress_components);
        Utilities::MPI::max (local_max_stress_components,
                             this->get_mpi_communicator(),
                             global_max_stress_components);
      }

      // finally produce something for the statistics file
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          statistics.add_value ("Minimal value for stress component " + this->introspection().name_for_compositional_index(c),
                                global_min_stress_components[c]);
          statistics.add_value ("Maximal value for stress component " + this->introspection().name_for_compositional_index(c),
                                global_max_stress_components[c]);
        }

      // also make sure that the other columns filled by this object
      // all show up with sufficient accuracy and in scientific notation
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          const std::string columns[] = { "Minimal value for composition " + this->introspection().name_for_compositional_index(c),
                                          "Maximal value for composition " + this->introspection().name_for_compositional_index(c)
                                        };
          for (unsigned int i=0; i<sizeof(columns)/sizeof(columns[0]); ++i)
            {
              statistics.set_precision (columns[i], 8);
              statistics.set_scientific (columns[i], true);
            }
        }

      std::ostringstream output;
      output.precision(4);
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          output << global_min_stress_components[c] << '/'
                 << global_max_stress_components[c];
          if (c+1 != this->n_compositional_fields())
            output << " // ";
        }

      return std::pair<std::string, std::string> ("Stress component min/max:",
                                                  output.str());
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(StressComponentStatistics,
                                  "stress component statistics",
                                  "A postprocessor that computes some statistics about "
                                  "the components of the stress tensor.")
  }
}
