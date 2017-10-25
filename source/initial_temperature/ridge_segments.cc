/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
 */


#include <aspect/initial_temperature/ridge_segments.h>
#include <aspect/geometry_model/box.h>
#include <aspect/initial_composition/ridge_segments.h>
#include <aspect/material_model/visco_plastic.h>

namespace aspect
{
  namespace InitialTemperature
  {
    template <int dim>
    RidgeSegments<dim>::RidgeSegments ()
    {}

    template <int dim>
    void
    RidgeSegments<dim>::
    initialize ()
    {
      // Check that the required material model ("visco plastic") is used
      AssertThrow((dynamic_cast<MaterialModel::ViscoPlastic<dim> *> (const_cast<MaterialModel::Interface<dim> *>(&this->get_material_model()))) != 0,
                  ExcMessage("The lithosphere with rift initial temperature plugin requires the viscoplastic material model plugin."));

      // Check that there are temperatues set at the top and bottom boundaries
      AssertThrow ((this->get_fixed_temperature_boundary_indicators().size() > 0)
                   &&
                   (this->get_boundary_temperature_manager().maximal_temperature(this->get_fixed_temperature_boundary_indicators())
                    !=
                    this->get_boundary_temperature_manager().minimal_temperature(this->get_fixed_temperature_boundary_indicators())),
                   ExcMessage("This plugin requires fixed top and bottom boundary temperatures."));

      // Convert spreading velocity in m/yr to m/s if needed
      if (this->convert_output_to_years())
        spreading_velocity /= year_in_seconds;
    }

    template <int dim>
    double
    RidgeSegments<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      // Determine coordinate system
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL ? true : false;

      double distance_to_ridge = 1e23;
      Point<2> surface_position;
      const std::list<std_cxx11::shared_ptr<InitialComposition::Interface<dim> > > initial_composition_objects = this->get_initial_composition_manager().get_active_initial_composition_conditions();
      for (typename std::list<std_cxx11::shared_ptr<InitialComposition::Interface<dim> > >::const_iterator it = initial_composition_objects.begin(); it != initial_composition_objects.end(); ++it)
        if ( InitialComposition::RidgeSegments<dim> *ic = dynamic_cast<InitialComposition::RidgeSegments<dim> *> ((*it).get()))
          {
            surface_position = ic->surface_position(position, cartesian_geometry);
            distance_to_ridge = ic->distance_to_ridge(surface_position, cartesian_geometry);
          }

      const double depth = this->get_geometry_model().depth(position);


                                                                                         // Determine plate age based on distance to the ridge
      const double plate_age = 0.5 * distance_to_ridge / spreading_velocity;

                                                                                         // The parameters needed for the plate cooling temperature calculation
                                                                                         const int n_sum = 100;
                                                                                         double sum = 0.0;

                                                                                         for (int i=1; i<=n_sum; i++)
      {
        sum += (1./i) *
                 (exp((-thermal_diffusivity*i*i*numbers::PI*numbers::PI*plate_age)/(max_plate_thickness*max_plate_thickness)))*
                 (sin(i*numbers::PI*depth/max_plate_thickness));
        }

      const double temperature = std::min(Tm,Ts + (Tm - Ts) * ((depth / max_plate_thickness) + (2.0 / numbers::PI) * sum));

                                 return temperature;
    }


    template <int dim>
    void
    RidgeSegments<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial temperature model");
      {
        prm.enter_subsection("Plate cooling");
        {
          prm.declare_entry ("Spreading velocity", "0.0407653503",
                             Patterns::Double (0),
                             "The spreading velocity of the MOR, used for the calculation "
                             "of the plate cooling model temperature. Units: m/years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "m/seconds otherwise.");
          prm.declare_entry ("Maximum oceanic plate thickness", "125000.0",
                             Patterns::Double (0),
                             "The maximum thickness of an oceanic plate in the plate cooling model "
                             "for when time goes to infinity. Units: m. " );
          prm.declare_entry ("Maximum oceanic plate temperature", "1593.00",
                             Patterns::Double (0),
                             "The maximum temperature of an oceanic plate in the plate cooling model "
                             "for when time goes to infinity. Units: K. " );
          prm.declare_entry ("Surface temperature", "273.00",
                             Patterns::Double (0),
                             "The fixed temperature at the top boundary of the model. "
                             "Units: K. " );
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    RidgeSegments<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Compositional fields");
      const unsigned int n_fields = prm.get_integer ("Number of fields");
      prm.leave_subsection ();
      prm.enter_subsection("Initial temperature model");
      {
        prm.enter_subsection("Plate cooling");
        {
          spreading_velocity = prm.get_double ("Spreading velocity");
          max_plate_thickness = prm.get_double ("Maximum oceanic plate thickness");
          Tm = prm.get_double ("Maximum oceanic plate temperature");
          Ts = prm.get_double ("Surface temperature");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

      AssertThrow(this->introspection().compositional_name_exists("crust"),
                  ExcMessage("We need a compositional field called 'crust' representing the oceanic crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_L"),
                  ExcMessage("We need a compositional field called 'mantle_L' representing the lithospheric part of the mantle."));
      const unsigned int id_mantle_L = this->introspection().compositional_index_for_name("mantle_L");

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Visco Plastic");
        {
          const std::vector<double> temp_thermal_diffusivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal diffusivities"))),
                                                                 n_fields+1,
                                                                 "Thermal diffusivities");

          // Assume the mantle lithosphere diffusivity is representative for whole plate
          thermal_diffusivity = temp_thermal_diffusivities[id_mantle_L+1];
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(RidgeSegments,
                                              "ridge segments",
                                              "An initial temperature field determined from the plate"
                                              "cooling model and mid ocean ridge segments. ")
  }
}
