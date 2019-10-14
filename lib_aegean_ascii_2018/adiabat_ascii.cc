/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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


#include "adiabat_ascii.h"
#include <aspect/initial_temperature/ascii_data.h>
#include <aspect/initial_composition/ascii_data.h>
#include <aspect/simulator_access.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>
//#include <aspect/geometry_model/layered_chunk_4.h>
#include "layered_chunk_4.h"
//#include "../lib_chunk/chunk_3.h"
//#include "../lib_chunk/chunk_3_layered.h"
#include "two_merged_chunks.h"
//#include "chunk_topo.h"

#include <cmath>

namespace aspect
{
  namespace InitialTemperature
  {
    template <int dim>
    double
    AdiabatAscii<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      AssertThrow ((dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()))
                   || (dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model())) != 0
//                   || (dynamic_cast<const GeometryModel::Chunk3<dim>*> (&this->get_geometry_model())) != 0
                   || (dynamic_cast<const GeometryModel::LayeredChunk4<dim>*> (&this->get_geometry_model())) != 0
 //                  || (dynamic_cast<const GeometryModel::ChunkTopo<dim>*> (&this->get_geometry_model())) != 0
                   || (dynamic_cast<const GeometryModel::TwoMergedChunks<dim>*> (&this->get_geometry_model())) != 0
                   || (dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model())) != 0,
                   ExcMessage ("This ascii data plugin can only be used when using "
                               "a spherical shell, chunk or box geometry."));

      const unsigned int n_compositional_fields = this->n_compositional_fields();

      // convert input ages to seconds
      const double age_top =    (this->convert_output_to_years() ? age_top_boundary_layer * year_in_seconds
                                 : age_top_boundary_layer);
      const double age_bottom = (this->convert_output_to_years() ? age_bottom_boundary_layer * year_in_seconds
                                 : age_bottom_boundary_layer);

      // First, get the temperature of the adiabatic profile at a representative
      // point at the top and bottom boundary of the model
      // if adiabatic heating is switched off, assume a constant profile
      const Point<dim> surface_point = this->get_geometry_model().representative_point(0.0);
      const Point<dim> bottom_point = this->get_geometry_model().representative_point(this->get_geometry_model().maximal_depth());
      const double adiabatic_surface_temperature = this->get_adiabatic_conditions().temperature(surface_point);
      const double adiabatic_bottom_temperature = (this->include_adiabatic_heating())
                                                  ?
                                                  this->get_adiabatic_conditions().temperature(bottom_point)
                                                  :
                                                  adiabatic_surface_temperature;

      // then, get the temperature at the top and bottom boundary of the model
      // if no boundary temperature is prescribed simply use the adiabatic.
      // This implementation assumes that the top and bottom boundaries have
      // prescribed temperatures and minimal_temperature() returns the value
      // at the surface and maximal_temperature() the value at the bottom.
      const double T_surface = (this->has_boundary_temperature()
                                ?
                                this->get_boundary_temperature().minimal_temperature(
                                  this->get_fixed_temperature_boundary_indicators())
                                :
                                adiabatic_surface_temperature);
      const double T_bottom = (this->has_boundary_temperature()
                               ?
                               this->get_boundary_temperature().maximal_temperature(
                                 this->get_fixed_temperature_boundary_indicators())
                               :
                               adiabatic_bottom_temperature);

      // get a representative profile of the compositional fields as an input
      // for the material model
    const double depth = this->get_geometry_model().depth(position);
//      double depth;
//      if (const GeometryModel::ChunkTopo<dim> *gm = dynamic_cast<const GeometryModel::ChunkTopo<dim> *>(&this->get_geometry_model()))
//        depth = gm->depth_wrt_topo(position);
//      else
//        depth = this->get_geometry_model().depth(position);

      // use correct depth
      std::array<double,dim> spherical_position;
      for (unsigned int d=0; d<dim; d++)
        spherical_position[d] = Utilities::Coordinates::cartesian_to_spherical_coordinates(position)[d];
      // TODO get run time outer radius
      spherical_position[0] = 6371000.0 - depth;

      const Point<dim> corrected_position = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(spherical_position);

      // look up material properties
      MaterialModel::MaterialModelInputs<dim> in(1, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(1, this->n_compositional_fields());
      in.position[0]=position;
      in.temperature[0]=this->get_adiabatic_conditions().temperature(corrected_position);
      in.pressure[0]=this->get_adiabatic_conditions().pressure(corrected_position);
      in.velocity[0]= Tensor<1,dim> ();
      for (unsigned int c=0; c<n_compositional_fields; ++c)
        in.composition[0][c] = this->get_initial_composition_manager().initial_composition(position, c);
      in.strain_rate.resize(0); // adiabat has strain=0.
      this->get_material_model().evaluate(in, out);

//      const double kappa = out.thermal_conductivities[0] / (out.densities[0] * out.specific_heat[0]);

      // analytical solution for the thermal boundary layer from half-space cooling model
      const double surface_cooling_temperature = age_top > 0.0 ?
                                                 (T_surface - adiabatic_surface_temperature) *
                                                 erfc(depth /
                                                      /*(2 * sqrt(kappa * age_top)))*/
                                                      (thickness_top))
                                                 : 0.0;
      const double bottom_heating_temperature = age_bottom > 0.0 ?
                                                (T_bottom - adiabatic_bottom_temperature + subadiabaticity)
                                                * erfc((this->get_geometry_model().maximal_depth()
                                                        - depth) /
                                                       /*(2 * sqrt(kappa * age_bottom)))*/
                                                       (thickness_bottom))
                                                : 0.0;

      // set the initial temperature perturbations based on an
      // ascii file with temperature anomalies computed from velocity anomalies
      // if wanted. Also smooth the perturbation to zero around a certain depth
      // at the top and/or bottom if needed. Also smooth towards the boundaries
      // and around the slab if necessary.

      const double centered_lon = (spherical_position[1] > numbers::PI) ? (360.0 - spherical_position[1] * 180.0 / numbers::PI) : spherical_position[1] * 180.0 / numbers::PI;
      const double centered_lat = 90.0 - spherical_position[2] * 180.0 / numbers::PI;
      const double perturbation = add_perturbation ?
                                  //(ascii_model->initial_temperature(corrected_position) *
                                  (this->get_initial_temperature_manager().template get_matching_initial_temperature_model<InitialTemperature::AsciiData<dim> >().initial_temperature(corrected_position) *
                                   //(100.0 *
                                   (add_top_smoothing ? 1.0-(0.5+0.5*std::tanh((top_smoothing_depth-depth)/top_smoothing_width)) : 1.0 ) *
                                   (add_bottom_smoothing ? (0.5+0.5*std::tanh((bottom_smoothing_depth-depth)/bottom_smoothing_width)) : 1.0 ) *
                                   (add_boundary_smoothing ?
                                    (0.5+0.5*std::tanh((boundary_smoothing_angle - std::max(std::abs(centered_lon),std::abs(centered_lat)))/boundary_smoothing_anglewidth))
                                    : 1.0) *
                                   (add_slab_smoothing ?
                                    std::max(1.0-(0.5+0.5*std::tanh((slab_smoothing_depth - depth)/slab_smoothing_width)),
                                             (1.0-(0.5+0.5*std::tanh((slab_smoothing_angle - std::max(std::abs(centered_lon),
                                                                      std::abs(centered_lat)))/slab_smoothing_anglewidth)))) : 1.0)) :
                                  0.0;

      // add the subadiabaticity
      double subadiabatic_T = 0.0;
      const double zero_depth = 0.174;
      const double nondimensional_depth = (depth / 2900000.0 - zero_depth)
                                          / (1.0 - zero_depth);
      if (nondimensional_depth > 0)
        subadiabatic_T = -subadiabaticity * nondimensional_depth * nondimensional_depth;

      // If adiabatic heating is disabled, apply all perturbations to
      // constant adiabatic surface temperature instead of adiabatic profile.
      const double temperature_profile = (this->include_adiabatic_heating())
                                         ?
                                         this->get_adiabatic_conditions().temperature(corrected_position)
                                         :
                                         adiabatic_surface_temperature;

      // return sum of the adiabatic profile, the boundary layer temperatures and the initial
      // temperature perturbation.
//      return temperature_profile + surface_cooling_temperature
//             + (perturbation > 0.0 ? std::max(bottom_heating_temperature + subadiabatic_T,perturbation)
//                : bottom_heating_temperature + subadiabatic_T);
      Assert(temperature_profile >= adiabatic_surface_temperature, ExcMessage("Adiabatic T profile lower than adiabatic surface T."));

//      if (subadiabatic_T != 0.0 || perturbation != 0.0)
//        std::cout << "subad T = " << subadiabatic_T << " perturb = " << perturbation << std::endl;

      return temperature_profile + surface_cooling_temperature + bottom_heating_temperature + subadiabatic_T + perturbation;
    }


    template <int dim>
    void
    AdiabatAscii<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial temperature model");
      {
        prm.enter_subsection("AdiabatAscii");
        {
          prm.declare_entry("Base model","ascii data",
                            Patterns::Selection("ascii data"),
                            "The name of a material model that will be modified by a depth "
                            "dependent viscosity. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Material models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.declare_entry ("Age top boundary layer", "0e0",
                             Patterns::Double (0),
                             "The age of the upper thermal boundary layer, used for the calculation "
                             "of the half-space cooling model temperature. Units: years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "seconds otherwise.");
          prm.declare_entry ("Thickness bottom boundary layer", "2e5",
                             Patterns::Double (0),
                             "The thickness of the lower thermal boundary layer, used for the calculation "
                             "of the half-space cooling model temperature. Units: meters. ");
          prm.declare_entry ("Thickness top boundary layer", "1e5",
                             Patterns::Double (0),
                             "The thickness of the upper thermal boundary layer, used for the calculation "
                             "of the half-space cooling model temperature. Units: meters ");
          prm.declare_entry ("Age bottom boundary layer", "0e0",
                             Patterns::Double (0),
                             "The age of the lower thermal boundary layer, used for the calculation "
                             "of the half-space cooling model temperature. Units: years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "seconds otherwise.");
          prm.declare_entry ("Add temperature perturbations", "true",
                             Patterns::Bool (),
                             "Whether or not temperature perturbations should be added to the "
                             "adiabatic profile with thermal boundary layers. ");
          prm.declare_entry ("Add smoothing to top perturbations", "false",
                             Patterns::Bool (),
                             "Whether or not smoothing of the temperature perturbations to zero should be "
                             "performed, or perturbations should be applied everywhere. ");
          prm.declare_entry ("Top smoothing depth", "2e5",
                             Patterns::Double (0),
                             "The depth around which the transition from applying temperature "
                             "perturbations to not applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Top smoothing width", "1e5",
                             Patterns::Double (0),
                             "The depth around which the transition from applying temperature "
                             "perturbations to not applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Add smoothing to bottom perturbations", "false",
                             Patterns::Bool (),
                             "Whether or not smoothing of the temperature perturbations to zero should be "
                             "performed, or perturbations should be applied everywhere. ");
          prm.declare_entry ("Bottom smoothing depth", "2.665e6",
                             Patterns::Double (0),
                             "The depth around which the transition from applying temperature "
                             "perturbations to not applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Bottom smoothing width", "1e5",
                             Patterns::Double (0),
                             "The depth around which the transition from applying temperature "
                             "perturbations to not applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Add smoothing to boundary perturbations", "false",
                             Patterns::Bool (),
                             "Whether or not smoothing of the temperature perturbations to zero should be "
                             "performed, or perturbations should be applied everywhere up to the domain boundaries. ");
          prm.declare_entry ("Boundary smoothing angle", "18",
                             Patterns::Double (0),
                             "The lat,lon angle from which the transition from applying temperature "
                             "perturbations to not applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Boundary smoothing angle width", "2",
                             Patterns::Double (0),
                             "The lat,lon angle from which the transition from applying temperature "
                             "perturbations to not applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Add smoothing to slab perturbations", "false",
                             Patterns::Bool (),
                             "Whether or not smoothing of the temperature perturbations to zero should be "
                             "performed, or perturbations should be applied everywhere, also in the slab. ");
          prm.declare_entry ("Slab smoothing angle", "2",
                             Patterns::Double (0),
                             "The angle around which the transition from applying no temperature "
                             "perturbations to applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Slab smoothing depth", "800000.0",
                             Patterns::Double (0),
                             "The depth around which the transition from applying no temperature "
                             "perturbations to applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Slab smoothing angle width", "1",
                             Patterns::Double (0),
                             "The angle around which the transition from applying no temperature "
                             "perturbations to applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Slab smoothing width", "100000.0",
                             Patterns::Double (0),
                             "The depth around which the transition from applying no temperature "
                             "perturbations to applying them occurs with a hyperbolical tangent. ");
          prm.declare_entry ("Subadiabaticity", "0e0",
                             Patterns::Double (0),
                             "If this value is larger than 0, the initial temperature profile will "
                             "not be adiabatic, but subadiabatic. This value gives the maximal "
                             "deviation from adiabaticity. Set to 0 for an adiabatic temperature "
                             "profile. Units: K.\n\n"
                             "The function object in the Function subsection "
                             "represents the compositional fields that will be used as a reference "
                             "profile for calculating the thermal diffusivity. "
                             "This function is one-dimensional and depends only on depth. The format of this "
                             "functions follows the syntax understood by the "
                             "muparser library, see Section~\\ref{sec:muparser-format}.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    AdiabatAscii<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial temperature model");
      {
        prm.enter_subsection("AdiabatAscii");
        {
          AssertThrow( prm.get("Base model") != "adiabatic profile with ascii perturbations",
                       ExcMessage("You may not use ``adiabatic profile with ascii perturbations'' as the base model for "
                                  "a this model model.") );

//          ascii_model.reset(InitialTemperature::create_initial_temperature_plugin<dim>(prm.get("Base model")));
//          if ( SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(ascii_model.get()))
//            sim->initialize_simulator (this->get_simulator());
          age_top_boundary_layer = prm.get_double ("Age top boundary layer");
          age_bottom_boundary_layer = prm.get_double ("Age bottom boundary layer");
          thickness_top= prm.get_double ("Thickness top boundary layer");
          thickness_bottom= prm.get_double ("Thickness bottom boundary layer");
          add_perturbation = prm.get_bool ("Add temperature perturbations");
          add_top_smoothing = prm.get_bool ("Add smoothing to top perturbations");
          top_smoothing_depth = prm.get_double ("Top smoothing depth");
          top_smoothing_width = prm.get_double ("Top smoothing width");
          add_bottom_smoothing = prm.get_bool ("Add smoothing to bottom perturbations");
          bottom_smoothing_depth = prm.get_double ("Bottom smoothing depth");
          bottom_smoothing_width = prm.get_double ("Bottom smoothing width");
          add_boundary_smoothing = prm.get_bool ("Add smoothing to boundary perturbations");
          boundary_smoothing_angle = prm.get_double ("Boundary smoothing angle");
          boundary_smoothing_anglewidth = prm.get_double ("Boundary smoothing angle width");
          add_slab_smoothing = prm.get_bool ("Add smoothing to slab perturbations");
          slab_smoothing_angle = prm.get_double ("Slab smoothing angle");
          slab_smoothing_depth = prm.get_double ("Slab smoothing depth");
          slab_smoothing_anglewidth = prm.get_double ("Slab smoothing angle width");
          slab_smoothing_width = prm.get_double ("Slab smoothing width");
          subadiabaticity = prm.get_double ("Subadiabaticity");

        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();


      // parse the ascii model parameters
      //ascii_model->parse_parameters(prm);
      //ascii_model->initialize();
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(AdiabatAscii,
                                       "adiabatic profile with ascii perturbations",
                                       "Temperature is prescribed as an adiabatic "
                                       "profile with upper and lower thermal boundary layers, "
                                       "whose ages are given as input parameters.")
  }
}
