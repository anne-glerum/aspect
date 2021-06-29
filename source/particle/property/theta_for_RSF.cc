/*
  Copyright (C) 2015 - 2020 by the authors of the ASPECT code.

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

#include <aspect/particle/property/theta_for_RSF.h>
#include <aspect/material_model/visco_plastic.h>
#include <aspect/initial_composition/interface.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      ThetaRSF<dim>::ThetaRSF ()
        :
        material_inputs(1,0),
        material_outputs(1,0)
      {}



      template <int dim>
      void
      ThetaRSF<dim>::initialize ()
      {
        material_inputs = MaterialModel::MaterialModelInputs<dim>(1, this->n_compositional_fields());

        material_outputs = MaterialModel::MaterialModelOutputs<dim>(1, this->n_compositional_fields());

        const MaterialModel::ViscoPlastic<dim> &viscoplastic
          = Plugins::get_plugin_as_type<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model());

        AssertThrow(viscoplastic.use_theta() == true,
                    ExcMessage ("The particle property 'theta' should only be used if a rate-and-state friction option is used"));

      }



      template <int dim>
      void
      ThetaRSF<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                      std::vector<double> &data) const
      {
        // Give theta its initial composition if one is prescribed.
        data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("theta")));
      }



      template <int dim>
      void
      ThetaRSF<dim>::update_particle_property(const unsigned int data_position,
                                              const Vector<double> &solution,
                                              const std::vector<Tensor<1,dim> > &gradients,
                                              typename ParticleHandler<dim>::particle_iterator &particle) const
      {
        AssertThrow(this->introspection().compositional_name_exists("fault"),
                    ExcMessage("Particle property theta for RSF only works if"
                               "there is a compositional particle field called fault."));
        const unsigned int initial_fault_idx = this->introspection().compositional_index_for_name("fault");
        const double initial_fault_value = solution[this->introspection().component_indices.compositional_fields[initial_fault_idx]];

        // only update theta if we are after time step 0, as currently we
        // do not have information about strain rate before updating the particle
        // it also only makes sense to update theta within the rate-and-state material, which currently must be called fault.
        if ((this->get_timestep_number() > 0)
            && (initial_fault_value > 0.5))
          {
            material_inputs.position[0] = particle->get_location();
            // ask for size of position! if it is 1 it is correct
            // print: material_inputs.position.size()

            material_inputs.current_cell  = typename DoFHandler<dim>::active_cell_iterator(*particle->get_surrounding_cell(this->get_triangulation()),
                                                                                           &(this->get_dof_handler()));

            // ToDo: Do I need to set temperature and pressure? THis is something that I copied from
            // the particle property elastic_stress, where it is set, but also not (visibly) used.
            material_inputs.temperature[0] = solution[this->introspection().component_indices.temperature];

            material_inputs.pressure[0] = solution[this->introspection().component_indices.pressure];

            for (unsigned int d = 0; d < dim; ++d)
              material_inputs.velocity[0][d] = solution[this->introspection().component_indices.velocities[d]];

            for (unsigned int n = 0; n < this->n_compositional_fields(); ++n)
              material_inputs.composition[0][n] = solution[this->introspection().component_indices.compositional_fields[n]];

            Tensor<2,dim> grad_u;
            for (unsigned int d=0; d<dim; ++d)
              grad_u[d] = gradients[d];
            material_inputs.strain_rate[0] = symmetrize (grad_u);

            // ToDo: find the problem with theta becoming negative/Zero and remove the next three lines and {}
            const MaterialModel::ViscoPlastic<dim> &viscoplastic
              = Plugins::get_plugin_as_type<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model());
            if (viscoplastic.use_print_thetas ())
              {
                const std::array<double,dim> coords = this->get_geometry_model().cartesian_to_natural_coordinates(material_inputs.position[0]);
                if (particle->get_properties()[data_position] < 0)
                  std::cout << "got a negative old     theta ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
                else if (particle->get_properties()[data_position] == 0)
                  std::cout << "got Zero old     theta ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
                else
                  std::cout << "got a positive old     theta on the particle ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
              }

            this->get_material_model().evaluate (material_inputs,material_outputs);
            // ToDo(?): should I directly call compute_theta instead of using the reaction terms?
            // Then I could use the old particle value directly as old theta without any averaging etc in  between
            // The tricky part would however be to get all the necessary parameters to compute edot_ii -> call material model with material model inputs to have all information -> make a new function to test it with input: material_inputs
            // TOdo: How dooes it work with the reaction terms and particles? Are the reaction terms computed for the element or for each particle?
            // Might make quite a difference here!
            particle->get_properties()[data_position] += material_outputs.reaction_terms[0][this->introspection().compositional_index_for_name("theta")];

            // ToDo: find the problem with theta becoming negative/Zero and remove the next three lines and {}
            if (viscoplastic.use_print_thetas ())
              {
                const std::array<double,dim> coords = this->get_geometry_model().cartesian_to_natural_coordinates(material_inputs.position[0]);
                if (particle->get_properties()[data_position] < 0)
                  std::cout << "got a negative current theta ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
                else if (particle->get_properties()[data_position] == 0)
                  std::cout << "got Zero current theta  ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
                else
                  std::cout << "got a positive current theta on the particle ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
              }

            // if theta got negative for whatever reason, set a positive value instead
            // ToDo: find out why this happens at all. Or more precisely: Why that explicitly positive value
            // can become negative once I read it in again in the next time step
            if (particle->get_properties()[data_position] < 1e-50)
            {
              particle->get_properties()[data_position] = 1e-50;
            {
              if (viscoplastic.use_print_thetas ())
                  {
                    //const std::array<double,dim> coords = this->get_geometry_model().cartesian_to_natural_coordinates(material_inputs.position[0]);
                    std::cout << "got a negative reset   theta ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
                  }
                else if (particle->get_properties()[data_position] == 0)
                  {
                    //const std::array<double,dim> coords = this->get_geometry_model().cartesian_to_natural_coordinates(material_inputs.position[0]);
                    std::cout << "got Zero reset   theta ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] << std::endl;
                  }
                 else
                   std::cout << "got a positive reset   theta on the particle ( "<<particle->get_properties()[data_position]<< " ) on the particle "<<particle->get_id() << " at dt "<< this->get_timestep_number() <<" in position (x-y-z): "<< coords[0]<< " -- "<< coords[1]<< " -- "<< coords[2] <<  std::endl;
           }
            }
          }
        else
          return;
      }



      template <int dim>
      UpdateTimeFlags
      ThetaRSF<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      UpdateFlags
      ThetaRSF<dim>::get_needed_update_flags () const
      {
        return update_values | update_gradients;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      ThetaRSF<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int> > property_information;

        //Check which fields are used in model and make an output for each.
        if (this->introspection().compositional_name_exists("theta"))
          property_information.emplace_back("theta",1);

        return property_information;
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(ThetaRSF,
                                        "theta for RSF",
                                        "A plugin in which the particle property is "
                                        "defined as the state variable theta in rate-and-"
                                        "state friction. See the friction rheology model "
                                        "documentation for more detailed information.")

    }
  }
}
