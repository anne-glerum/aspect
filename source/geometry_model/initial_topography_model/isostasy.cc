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


#include <aspect/geometry_model/initial_topography_model/isostasy.h>
#include <aspect/initial_composition/litho1.h>
#include <aspect/geometry_model/box.h>
#include <aspect/utilities.h>
#include <boost/lexical_cast.hpp>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/std_cxx11/array.h>

namespace aspect
{
  namespace InitialTopographyModel
  {
    template <int dim>
    Isostasy<dim>::Isostasy ()
      :
      surface_boundary_id(1),
      bottom_boundary_id(0)
    {}


    template <int dim>
    void
    Isostasy<dim>::
    initialize ()
    {
      std::cout << "Initializing topo " << std::endl;
      // Find the boundary indicators that represents the surface and the bottom of the domain
      surface_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      // Abuse the top and bottom boundary id to create to tables,
      // one for the crustal thickness, and one for the mantle thickness
      // surface_boundary_id indicates the crust table and
      // bottom_boundary_id the LAB table
      std::set<types::boundary_id> boundary_set;
      boundary_set.insert(surface_boundary_id);
      boundary_set.insert(bottom_boundary_id);

      // The input ascii table contains one component, either the crust depth or the LAB depth
      Utilities::AsciiDataBoundary<dim>::initialize(boundary_set,
                                                    1);

      // Compute the maximum topography amplitude based on isostasy.
      // Assume the reference density is representative for each layer (despite temperature dependence)
      AssertThrow(this->introspection().compositional_name_exists("upper"),ExcMessage("We need a compositional field called 'upper' representing the upper crust."));
      AssertThrow(this->introspection().compositional_name_exists("lower"),ExcMessage("We need a compositional field called 'lower' representing the lower crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_L"),ExcMessage("We need a compositional field called 'mantle_L' representing the lithospheric part of the mantle."));
      std::cout << "Getting compo indices" << std::endl;
      const unsigned int id_upper = this->introspection().compositional_index_for_name("upper");
      const unsigned int id_lower = this->introspection().compositional_index_for_name("lower");
      const unsigned int id_mantle_L = this->introspection().compositional_index_for_name("mantle_L");

      // Take the densities we need
      // Account for the additional field for the residual mantle
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

      // Add sublithospheric mantle part to the columns
      ref_rgh += (compensation_depth - sum_thicknesses) * densities[0];
    }


    template <int dim>
    double
    Isostasy<dim>::
    value (const Point<dim-1> &surface_position) const
    {
      // Make position dim-dimensional again
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL ? true : false;
      Point<dim> position;
      if (cartesian_geometry)
        for (unsigned int d=0; d<dim-1; ++d)
          position[d] = surface_position[d];
      else
        {
          std_cxx11::array<double,dim> spherical_position;
          for (unsigned int d=1; d<dim; ++d)
            spherical_position[d] = surface_position[d-1]/180.*numbers::PI;
          spherical_position[0] = 6371000.;
          position = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(spherical_position);
        }

      // The Moho depth is the first component
      const double Moho_depth = Utilities::AsciiDataBoundary<dim>::get_data_component(surface_boundary_id,
                                                                                      position,
                                                                                      0);

      // The LAB depth is the second component
      const double LAB_depth = std::max(Moho_depth, Utilities::AsciiDataBoundary<dim>::get_data_component(bottom_boundary_id,
                                                                                                          position,
                                                                                                          0));

      const double upper_crust_depth = upper_crust_fraction * Moho_depth;

      // The local lithospheric column
      double local_rgh = densities[1]*upper_crust_depth + densities[2]*(Moho_depth-upper_crust_depth)+densities[3]*(LAB_depth-Moho_depth);

      // The total local lithosphere thickness
      const double sum_local_thicknesses = LAB_depth;
      local_rgh += (compensation_depth - sum_local_thicknesses) * densities[0];

      return (ref_rgh - local_rgh) / densities[0];
    }


    template <int dim>
    double
    Isostasy<dim>::
    max_topography () const
    {
      return 0;
    }


    template <int dim>
    void
    Isostasy<dim>::
    declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry model");
      {
        prm.enter_subsection("Initial topography model");
        {
          //Utilities::AsciiDataBoundary<dim>::declare_parameters(prm,
          // Use AsciiDataBase instead of AsciiDataBoundary, because
          // the latter requires general variable information (use_years_instead_of_seconds)
          // that is not available yet at the time of parsing initial topography parameters.
          // Moreover, as there is no time-dependence for the 'initial' topography,
          // we don't need these parameters anyway.
          Utilities::AsciiDataBase<dim>::declare_parameters(prm,
                                                            "$ASPECT_SOURCE_DIR/data/initial-temperature/ascii-data/test/",
                                                            "box_2d_%s.%d.txt");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    Isostasy<dim>::parse_parameters (ParameterHandler &prm)
    {
      std::cout << "Param topo " << std::endl;
      unsigned int n_fields;
      prm.enter_subsection ("Compositional fields");
      {
        n_fields = prm.get_integer ("Number of fields");
      }
      prm.leave_subsection();

      std::cout << "Param topo 1" << std::endl;

      prm.enter_subsection("Geometry model");
      {
        prm.enter_subsection("Initial topography model");
        {
          std::cout << "Param topo 1a" << std::endl;
          //Utilities::AsciiDataBoundary<dim>::parse_parameters(prm);
          Utilities::AsciiDataBase<dim>::parse_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection ("Initial composition model");
      {
        prm.enter_subsection("LITHO1.0");
        {
          upper_crust_fraction = prm.get_double ("Upper crust fraction");
          std::cout << "Param topo 1b" << std::endl;
          thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Layer thicknesses"))),
                                                                3,
                                                                "Layer thicknesses");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      std::cout << "Param topo 2" << std::endl;

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

      std::cout << "Param topo 3" << std::endl;

      prm.enter_subsection ("Initial temperature model");
      {
        prm.enter_subsection ("Litho1.0");
        {
          compensation_depth = prm.get_double ("Temperature compensation depth");
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
    ASPECT_REGISTER_INITIAL_TOPOGRAPHY_MODEL(Isostasy,
                                             "isostasy",
                                             "An initial topography model that defines the initial topography "
                                             "as constant inside each of a set of polylineal parts of the "
                                             "surface. The polylines, and their associated surface elevation, "
                                             "are defined in the `Geometry model/Initial topography/Prm polyline' "
                                             "section.")
  }
}
