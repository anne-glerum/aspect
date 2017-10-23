/*
  Copyright (C) 2016 by the authors of the ASPECT code.

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


#include <aspect/geometry_model/initial_topography_model/lithosphere_rift.h>
#include <aspect/initial_composition/lithosphere_rift.h>
#include <aspect/geometry_model/box.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/utilities.h>
#include <boost/lexical_cast.hpp>

namespace aspect
{
  namespace InitialTopographyModel
  {
    template <int dim>
    void
    LithosphereRift<dim>::
    initialize ()
    {
      // Check that the required initial composition model is used
//      const std::vector<std::string> active_initial_composition_models = this->get_initial_composition_manager().get_active_initial_composition_names();
//      AssertThrow(find(active_initial_composition_models.begin(),active_initial_composition_models.end(), "lithosphere with rift") != active_initial_composition_models.end(),
//                  ExcMessage("The lithosphere with rift initial topography plugin requires the lithosphere with rift initial composition plugin."));

      // Compute the maximum topography amplitude based on isostasy.
      // Assume the reference density is representative for each layer (despite temperature dependence)

      // For now, we assume a 3-layer system with an upper crust, lower crust and lithospheric mantle
      const unsigned int id_upper = this->introspection().compositional_index_for_name("upper");
      const unsigned int id_lower = this->introspection().compositional_index_for_name("lower");
      const unsigned int id_mantle_L = this->introspection().compositional_index_for_name("mantle_L");

      densities.push_back(temp_densities[0]);
      densities.push_back(temp_densities[id_upper+1]);
      densities.push_back(temp_densities[id_lower+1]);
      densities.push_back(temp_densities[id_mantle_L+1]);

      // The reference column
      ref_rgh = 0;
      // Assume constant gravity magnitude, so ignore
      for (unsigned int l=0; l<3; ++l)
        ref_rgh += densities[l+1] * thicknesses[l];

      // The total lithosphere thickness
      const double sum_thicknesses = std::accumulate(thicknesses.begin(), thicknesses.end(),0);

      // Make sure the compensation depth is in the sublithospheric mantle
      compensation_depth = 2. * sum_thicknesses + 1e3;

      // The column at the rift center
      double rift_rgh = 0;
      rift_thicknesses = thicknesses;
      for (unsigned int l=0; l<rift_thicknesses.size(); ++l)
        rift_thicknesses[l] *= (1.-A);

       for (unsigned int l=0; l<3; ++l)
         rift_rgh += densities[l+1] * rift_thicknesses[l];

       // The total lithosphere thickness at the rift
       const double sum_rift_thicknesses = sum_thicknesses * (1.-A);

       // The column at the polygon center
       double polygon_rgh = 0;
       for (unsigned int l=0; l<3; ++l)
         polygon_rgh += densities[l+1] * polygon_thicknesses[l];

       // The total lithosphere thickness
       const double sum_polygon_thicknesses = std::accumulate(polygon_thicknesses.begin(), polygon_thicknesses.end(),0);

      // Add sublithospheric mantle part to the columns
      ref_rgh += (compensation_depth - sum_thicknesses) * densities[0];
      rift_rgh += (compensation_depth - sum_rift_thicknesses) * densities[0];
      polygon_rgh += (compensation_depth - sum_polygon_thicknesses) * densities[0];

      // Compute the maximum topography based on mass surplus/deficit
      topo_rift_amplitude = (ref_rgh-rift_rgh) / densities[0];
      topo_polygon_amplitude = (ref_rgh-polygon_rgh) / densities[0];

      this->get_pcout() << "   Maximum initial topography of rift: " << topo_rift_amplitude << " m" << std::endl;
      this->get_pcout() << "   Maximum initial topography of polygon: " << topo_polygon_amplitude << " m" << std::endl;
    }


    template <int dim>
    double
    LithosphereRift<dim>::
    value (const Point<dim-1> &position) const
    {
      // Determine coordinate system
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL ? true : false;

      // Get the distance to the line segments along a path parallel to the surface
      double distance_to_rift_axis = 1e23;
      Point<2> surface_position;
      // Convert to radians and colat
      for (unsigned int d=0; d<dim-1; ++d)
        surface_position[d] = position[d]*numbers::PI/180.;
      surface_position[1] = 0.5*numbers::PI - surface_position[1];
      double distance_to_L_polygon = 1e23;
      const std::list<std_cxx11::shared_ptr<InitialComposition::Interface<dim> > > initial_composition_objects = this->get_initial_composition_manager().get_active_initial_composition_conditions();
      for (typename std::list<std_cxx11::shared_ptr<InitialComposition::Interface<dim> > >::const_iterator it = initial_composition_objects.begin(); it != initial_composition_objects.end(); ++it)
        if( InitialComposition::LithosphereRift<dim> *ic = dynamic_cast<InitialComposition::LithosphereRift<dim> *> ((*it).get()))
          {
            distance_to_rift_axis = ic->distance_to_rift(surface_position, cartesian_geometry);
            distance_to_L_polygon = ic->distance_to_polygon(surface_position);
          }

      // Compute the topography based on distance to the rift
      //return topo_rift_amplitude * std::exp(-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)));

      // Compute the topography based on distance to the rift and distance to the polygon
      std::vector<double> local_thicknesses(3);
      local_thicknesses[0] = ((0.5+0.5*std::tanh(distance_to_L_polygon/sigma))*polygon_thicknesses[0]+(0.5-0.5*std::tanh(distance_to_L_polygon/sigma))*thicknesses[0])*
          (1.0 - A * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)))));
      local_thicknesses[0] = ((0.5+0.5*std::tanh(distance_to_L_polygon/sigma))*polygon_thicknesses[1]+(0.5-0.5*std::tanh(distance_to_L_polygon/sigma))*thicknesses[1])*
          (1.0 - A * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)))));
      local_thicknesses[0] = ((0.5+0.5*std::tanh(distance_to_L_polygon/sigma))*polygon_thicknesses[2]+(0.5-0.5*std::tanh(distance_to_L_polygon/sigma))*thicknesses[2])*
          (1.0 - A * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)))));

      // The local lithospheric column
      double local_rgh = 0;
      for (unsigned int l=0; l<3; ++l)
        local_rgh += densities[l+1] * local_thicknesses[l];
      // The total local lithosphere thickness
      const double sum_local_thicknesses = std::accumulate(local_thicknesses.begin(), local_thicknesses.end(),0);
      local_rgh += (compensation_depth - sum_local_thicknesses) * densities[0];

      return (ref_rgh - local_rgh) / densities[0];
    }



    template <int dim>
    void
    LithosphereRift<dim>::
    declare_parameters (ParameterHandler &)
    {
    }



    template <int dim>
    void
    LithosphereRift<dim>::parse_parameters (ParameterHandler &prm)
    {
      unsigned int n_fields;
      prm.enter_subsection ("Compositional fields");
      {
       n_fields = prm.get_integer ("Number of fields");
      }
      prm.leave_subsection();
      prm.enter_subsection ("Initial composition model");
      {
        prm.enter_subsection("Lithosphere with rift");
        {
          sigma                = prm.get_double ("Standard deviation of Gaussian noise amplitude distribution");
          A                    = prm.get_double ("Maximum amplitude of Gaussian noise amplitude distribution");
          thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Layer thicknesses"))),
                                                                3,
                                                                "Layer thicknesses");
          polygon_thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Lithospheric polygon layer thicknesses"))),
                                                                        3,
                                                                        "Lithospheric polygon layer thicknesses");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();


      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Visco Plastic");
        {
          // The material model viscoplastic prefixes an entry for the background material
         temp_densities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Densities"))),
                                                             n_fields+1,
                                                             "Densities");
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
  namespace InitialTopographyModel
  {
    ASPECT_REGISTER_INITIAL_TOPOGRAPHY_MODEL(LithosphereRift,
                                             "lithosphere with rift",
                                             "An initial topography model that defines the initial topography "
                                             "as constant inside each of a set of polylineal parts of the "
                                             "surface. The polylines, and their associated surface elevation, "
                                             "are defined in the `Geometry model/Initial topography/Prm polyline' "
                                             "section.")
  }
}
