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
      // Find the boundary indicators that represents the surface and the bottom of the domain
      surface_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");
      left_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("west");

      // Abuse the top and bottom boundary id to create to tables,
      // one for the crustal thickness, and one for the mantle thickness
      // surface_boundary_id indicates the crust table and
      // bottom_boundary_id the LAB table
      std::set<types::boundary_id> boundary_set;
      boundary_set.insert(surface_boundary_id);
      boundary_set.insert(bottom_boundary_id);
      boundary_set.insert(left_boundary_id);

      // The input ascii table contains one component, either the crust depth or the LAB depth
      Utilities::AsciiDataBoundary<dim>::initialize(boundary_set,
                                                    1);

      // Compute the maximum topography amplitude based on isostasy.
      // Assume the reference density is representative for each layer (despite temperature dependence)
      AssertThrow(this->introspection().compositional_name_exists("upper"),ExcMessage("We need a compositional field called 'upper' representing the upper crust."));
      AssertThrow(this->introspection().compositional_name_exists("lower"),ExcMessage("We need a compositional field called 'lower' representing the lower crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_L"),ExcMessage("We need a compositional field called 'mantle_L' representing the lithospheric part of the mantle."));

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
        {
        for (unsigned int d=0; d<dim-1; ++d)
          position[d] = surface_position[d];
        }
      else
        {
          std_cxx11::array<double,dim> spherical_position;
          for (unsigned int d=1; d<dim; ++d)
            spherical_position[d] = surface_position[d-1]/180.*numbers::PI;
          // set radius to surface radius
          spherical_position[0] = 6371000.;
          // convert from latitude to colatitude
          spherical_position[dim-1] = 0.5*numbers::PI-spherical_position[dim-1];
          position = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(spherical_position);
        }

      // The Moho depth is stored in the surface file
      const double Moho_depth = Utilities::AsciiDataBoundary<dim>::get_data_component(surface_boundary_id,
                                                                                      position,
                                                                                      0);

      // The LAB depth is the second and third component
      const double LAB_depth_1 = std::max(min_LAB_thickness,
                                        std::max(Moho_depth,
                                                 Utilities::AsciiDataBoundary<dim>::get_data_component(bottom_boundary_id,
                                                                                                       position,
                                                                                                       0)));

      const double LAB_depth_2 = std::max(min_LAB_thickness,
                                        std::max(Moho_depth,
                                                 Utilities::AsciiDataBoundary<dim>::get_data_component(left_boundary_id,
                                                                                                       position,
                                                                                                       0)));

      // Inside is positive, outside negative.
      double distance_to_polygon = 0;
      if (dim == 2)
        {
          double sign = -1.;
          if (surface_position[0]>polygon_point_list[0][0] && surface_position[0]<polygon_point_list[1][0])
            sign = 1.;
          distance_to_polygon = sign * std::min(std::abs(polygon_point_list[1][0] - surface_position[0]), std::abs(surface_position[0] - polygon_point_list[0][0]));
        }
      else
        {
          distance_to_polygon = Utilities::signed_distance_to_polygon<dim>(polygon_point_list, Point<2>(surface_position[0],surface_position[dim-2]));
        }

      const double LAB_depth =  merge_LAB_grids ? (0.5+0.5*std::tanh(distance_to_polygon/merge_LAB_grids_halfwidth))*LAB_depth_2
                                +(0.5-0.5*std::tanh(distance_to_polygon/merge_LAB_grids_halfwidth))*LAB_depth_1
                                : LAB_depth_1;

      const double upper_crust_depth = upper_crust_fraction * Moho_depth;

      // The local lithospheric column
      const double sum_local_thicknesses = LAB_depth;
      const double local_rgh = densities[1]*upper_crust_depth + densities[2]*(Moho_depth-upper_crust_depth)+densities[3]*(LAB_depth-Moho_depth)
                         + (compensation_depth - sum_local_thicknesses) * densities[0];

      return (ref_rgh - local_rgh) / densities[0];
    }


    template <int dim>
    double
    Isostasy<dim>::
    max_topography () const
    {
      return -7060.61;
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
      unsigned int n_fields;
      prm.enter_subsection ("Compositional fields");
      {
        n_fields = prm.get_integer ("Number of fields");
      }
      prm.leave_subsection();

      prm.enter_subsection("Geometry model");
      {
        prm.enter_subsection("Initial topography model");
        {
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
          thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Layer thicknesses"))),
                                                                3,
                                                                "Layer thicknesses");
          min_LAB_thickness = prm.get_double ("Minimum LAB thickness");
          merge_LAB_grids   = prm.get_bool ("Merge LAB grids");
          merge_LAB_grids_halfwidth = prm.get_double ("LAB grid merge halfwidth");

           // Split the string into point strings
           const std::vector<std::string> temp_points = Utilities::split_string_list(prm.get("LAB grid merge polygon"),'>');
           const unsigned int n_temp_points = temp_points.size();
           if (dim == 3)
             {
               AssertThrow(n_temp_points>=3, ExcMessage ("The number of polygon points should be equal to or larger than 3 in 3d."));
             }
           else
             {
               AssertThrow(n_temp_points==2, ExcMessage ("The number of polygon points should be equal to 2 in 2d."));
             }
           polygon_point_list.resize(n_temp_points);
           // Loop over the points of the polygon.
           for (unsigned int i_points = 0; i_points < n_temp_points; i_points++)
             {
               const std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_points[i_points],','));
               Assert(temp_point.size() == dim-1,ExcMessage ("The given coordinates of point '" + temp_points[i_points] + "' are not correct. "
                                                             "It should only contain 1 (2d) or 2 (in 3d) parts: "
                                                             "the longitude/x (and latitude/y in 3d) coordinate (separated by a ',')."));

               // Add the point to the list of points for this segment
               polygon_point_list[i_points][0] = temp_point[0];
               polygon_point_list[i_points][1] = temp_point[dim-2];
             }
           if  (dim == 2)
             AssertThrow(polygon_point_list[0][0] < polygon_point_list[1][0], ExcMessage("The order of the x coordinates of the 2 points "
                 "of each 2d polygon should be ascending. "));

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
