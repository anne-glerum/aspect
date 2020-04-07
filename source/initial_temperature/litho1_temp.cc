/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

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


#include <aspect/global.h>
#include <aspect/initial_temperature/litho1_temp.h>
#include <aspect/heating_model/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/material_model/visco_plastic_strain.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/initial_topography_model/interface.h>

namespace aspect
{
  namespace InitialTemperature
  {
    template <int dim>
    Litho1<dim>::Litho1 ()
      :
      surface_boundary_id(1)
    {}


    template <int dim>
    void
    Litho1<dim>::initialize ()
    {
      // Find the boundary indicators that represents the surface and the bottom of the domain
      surface_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");
      left_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("left");

      // Abuse the top and bottom boundary id to create to tables,
      // one for the crustal thickness, and one for the mantle thickness
      // surface_boundary_id indicates the crust table and
      // bottom_boundary_id the LAB table
      std::set<types::boundary_id> boundary_set;
      boundary_set.insert(surface_boundary_id);
      boundary_set.insert(bottom_boundary_id);
      boundary_set.insert(left_boundary_id);

      // The input ascii table contains one component, either the crust height or the LAB height
      Utilities::AsciiDataBoundary<dim>::initialize(boundary_set,
                                                    1);

      // Check that the required radioactive heating model ("compositional heating") is used
      const std::vector<std::string> &heating_models = this->get_heating_model_manager().get_active_heating_model_names();
      AssertThrow(std::find(heating_models.begin(), heating_models.end(), "compositional heating") != heating_models.end(),
                  ExcMessage("The continental geotherm initial temperature plugin requires the compositional heating plugin."));

      // Check that the required material model ("visco plastic") is used
      AssertThrow((dynamic_cast<MaterialModel::ViscoPlasticStrain<dim> *> (const_cast<MaterialModel::Interface<dim> *>(&this->get_material_model()))) != 0,
                  ExcMessage("The continental geotherm initial temperature plugin requires the viscoplastic material model plugin."));
      // Check that we're using the box geometry model as not all functionality is implemented for spherical as well
      AssertThrow((dynamic_cast<GeometryModel::Box<dim> *> (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model()))) != 0,
                  ExcMessage("The continental geotherm initial temperature plugin only works with the box geometry model plugin."));
    }


    template <int dim>
    double
    Litho1<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      // We want to get at the Moho heigth, which is the first component
      const double UC_radius = Utilities::AsciiDataBoundary<dim>::get_data_component(surface_boundary_id,
                                                                                     position,
                                                                                     0);
      // The LAB height is the second and third component
      const double Moho_radius =
        std::min(UC_radius,
                 Utilities::AsciiDataBoundary<dim>::get_data_component(bottom_boundary_id,
                                                                       position,
                                                                       0));


      Point<dim-1> surface_position;
      double radius = 0;
      double reference_radius = 0;
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL;
      if (cartesian_geometry)
        {
          for (unsigned int d=0; d<dim-1; ++d)
            surface_position[d]=position[d];
          radius = position[dim-1];
          const GeometryModel::Box<dim> *gm = dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model());
          const double extent = gm->get_extents()[dim-1];
          const double origin = gm->get_origin()[dim-1];
          reference_radius = extent+origin;
        }
      // chunk (spherical) geometries
      else
        {
          // spherical coordinates in radius [m], lon [rad], colat [rad] format
          const std_cxx11::array<double,dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
          // return lon [degrees], lat [degrees]
          for (unsigned int d=0; d<dim-1; ++d)
            surface_position[d] = spherical_point[d+1]*180./numbers::PI;
          if (dim == 3)
            surface_position[dim-2] = 90. - surface_position[dim-2];
          radius = spherical_point[0];

          if (GeometryModel::SphericalShell<dim> *gm = dynamic_cast<GeometryModel::SphericalShell<dim> *>
                                                       (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
            {
              reference_radius = gm->outer_radius();
            }
          else if (GeometryModel::Chunk<dim> *gm = dynamic_cast<GeometryModel::Chunk<dim> *>
                                                   (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
            {
              reference_radius = gm->outer_radius();
            }
          else
            AssertThrow(false, ExcMessage("This geometry model is not compatible with the litho1_temp plugin"));
        }

      // Trick the Utilities fuction by switching coordinates
      // otherwise it will take the x coordinate for comparison with the table coordinates (y)
      // TODO spherical
      Point<dim> tmp_position = position;
      tmp_position[0]=0;
      for (unsigned int d=0; d<dim-1; ++d)
        tmp_position[d+1]=surface_position[d];
      const double LAB_radius =
        std::min(Moho_radius,
                 Utilities::AsciiDataBoundary<dim>::get_data_component(left_boundary_id,
                                                                       tmp_position,
                                                                       0));
      // Get pointer to initial topography model to compute the layer thicknesses
      //InitialTopographyModel::Interface<dim> *topo_model = const_cast<InitialTopographyModel::Interface<dim>*>(&this->get_initial_topography_model());
      //const double topo = topo_model->value(surface_position);
      const double topo = this->get_initial_topography_model().value(surface_position);
      const double topo_radius = reference_radius + ((this->get_pre_refinement_step()==this->get_parameters().initial_adaptive_refinement) ? topo : 0);
      if (this->get_pre_refinement_step()==this->get_parameters().initial_adaptive_refinement)
        {
          AssertThrow(topo_radius > 0., ExcMessage("domain radius smaller than zero"));
          AssertThrow(topo_radius-UC_radius >= 0., ExcMessage("domain radius smaller than UC_radius: " + std::to_string(topo_radius) + " <= " + std::to_string(UC_radius)));
        }
      else
        {
          this->get_pcout() << "Not in last AMR step, returning LAB isotherm" << std::endl;
          return LAB_isotherm;
        }
      AssertThrow(UC_radius-Moho_radius >= 0., ExcMessage("UC radius smaller than Moho_radius" + std::to_string(UC_radius) + " <= " + std::to_string(Moho_radius)));
      AssertThrow(Moho_radius-LAB_radius >= 0., ExcMessage("Moho radius smaller than LAB_radius"));
      const std::vector<double> thicknesses = {topo_radius-UC_radius, UC_radius-Moho_radius, Moho_radius-LAB_radius};

      // In the lithosphere, return a continental geotherm
      // that incorporates radioactive heating.
      //AssertThrow(topo_radius>=radius, ExcMessage("The current point has a larger radius than the domain"));
      const double depth = std::max(0.,topo_radius-radius);

      if (radius >= LAB_radius)
        {
          //return continental_geotherm(depth, thicknesses);
          // deal with small roundoff errors etc first
          const double tmpT = continental_geotherm(depth, thicknesses);
          //AssertThrow(tmpT >= T0, ExcMessage("continental T is too small " + std::to_string(tmpT) + " at depth " + std::to_string(depth) + " and height " + std::to_string(radius) + " for layer thicknesses " + std::to_string(thicknesses[0]) + ", " + std::to_string(thicknesses[1]) + ", " + std::to_string(thicknesses[2]) + " at x-coord " + std::to_string(position[0])));
          if (tmpT < T0)
            {
              if (tmpT<T0-5.)
                {
                  AssertThrow(false, ExcMessage("Continental T is too small " + std::to_string(tmpT) + " at depth " + std::to_string(depth) + " and height " + std::to_string(radius) + " for layer thicknesses " + std::to_string(thicknesses[0]) + ", " + std::to_string(thicknesses[1]) + ", " + std::to_string(thicknesses[2]) + " at x-coord " + std::to_string(position[0])));
                }
              else
                {
                  this->get_pcout() << "Continental T " << tmpT << " is slightly smaller than the surface T " << T0 << " at depth " << depth << " and height " << radius << " for layer thicknesses " << thicknesses[0] << ", " << thicknesses[1] << ", " << thicknesses[2] <<  " at x-coord " <<  position[0] << ". T is set to the surface T." << std::endl;
                  return T0;
                }
            }
          AssertThrow(tmpT < LAB_isotherm+50, ExcMessage("continental T is too big" + std::to_string(tmpT)));
          return tmpT;
        }
      // Up to a compensation depth, return the LAB temperature
      // This ensures there are no lateral temperature gradients
      // in the mantle
      else if (radius >= reference_radius-compensation_depth)
        return LAB_isotherm;
      // Return the adiabatic temperature computed in the adiabatic conditions plugin
      // First compute the z-position for a geometry without topography
      else
        {
          // TODO implement for spherical geometries or use new depth function
          // w.r.t. initial topo
          Point<dim> old_position = position;
          if (cartesian_geometry)
            {
              const double old_radius = radius / (1. + topo/reference_radius);
              old_position[dim-1] = old_radius;
            }
          else
            AssertThrow(false, ExcMessage("This geometry model is not compatible with the litho1_temp plugin"));

          return  this->get_adiabatic_conditions().temperature(old_position);
        }
    }

    template <int dim>
    double
    Litho1<dim>::
    continental_geotherm (const double depth,
                          const std::vector<double> layer_thicknesses) const
    {
      // Compute some constants
      const double a = 0.5*densities[0]*heat_productivities[0]*layer_thicknesses[0] + 0.5*densities[1]*heat_productivities[1]*layer_thicknesses[1] + conductivities[0]/layer_thicknesses[0]*T0;
      const double b = 1./(conductivities[0]/layer_thicknesses[0]+conductivities[1]/layer_thicknesses[1]);
      const double c = 0.5*densities[1]*heat_productivities[1]*layer_thicknesses[1] + conductivities[2]/layer_thicknesses[2]*LAB_isotherm;
      const double d = 1./(conductivities[1]/layer_thicknesses[1]+conductivities[2]/layer_thicknesses[2]);

      // Temperature at boundary between layer 1 and 2
      const double T1 = (a*b + conductivities[1]/layer_thicknesses[1]*c*d*b) / (1.-(conductivities[1]*conductivities[1])/(layer_thicknesses[1]*layer_thicknesses[1])*d*b);
      // Temperature at boundary between layer 2 and 3
      const double T2 = (c + conductivities[1]/layer_thicknesses[1]*T1) * d;

      // Temperature in layer 1
      if (depth < layer_thicknesses[0])
        return -0.5*densities[0]*heat_productivities[0]/conductivities[0]*std::pow(depth,2) + (0.5*densities[0]*heat_productivities[0]*layer_thicknesses[0]/conductivities[0] + (T1-T0)/layer_thicknesses[0])*depth + T0;
      // Temperature in layer 2
      else if (depth < layer_thicknesses[0]+layer_thicknesses[1])
        return -0.5*densities[1]*heat_productivities[1]/conductivities[1]*std::pow(depth-layer_thicknesses[0],2.) + (0.5*densities[1]*heat_productivities[1]*layer_thicknesses[1]/conductivities[1] + (T2-T1)/layer_thicknesses[1])*(depth-layer_thicknesses[0]) + T1;
      // Temperature in layer 3
      else if (depth <= layer_thicknesses[0]+layer_thicknesses[1]+layer_thicknesses[2])
        return (LAB_isotherm-T2)/layer_thicknesses[2] *(depth-layer_thicknesses[0]-layer_thicknesses[1]) + T2;
      // Return a constant sublithospheric temperature of 10*LAB_isotherm just in case
      else
        return 10.*LAB_isotherm;

    }

    template <int dim>
    void
    Litho1<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Initial temperature model");
      {
        Utilities::AsciiDataBoundary<dim>::declare_parameters(prm,
                                                              "$ASPECT_SOURCE_DIR/data/initial-temperature/ascii-data/test/",
                                                              "box_2d_%s.%d.txt");
        prm.enter_subsection ("Litho1.0");
        {
          prm.declare_entry ("LAB isotherm temperature", "1673.15",
                             Patterns::Double (0),
                             "The value of the isothermal boundary temperature assumed at the LAB "
                             "and up to the reference depth . Units: Kelvin.");
          prm.declare_entry ("Surface temperature", "273.15",
                             Patterns::Double (0),
                             "The value of the surface temperature. Units: Kelvin.");
          prm.declare_entry ("Temperature compensation depth", "200000.",
                             Patterns::Double (0),
                             "The depth to which the LAB isotherm is prescribed in case "
                             "the depth of the LAB is less than this depth. Units: m.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    Litho1<dim>::parse_parameters (ParameterHandler &prm)
    {
      unsigned int n_fields = 0;
      prm.enter_subsection ("Compositional fields");
      {
        n_fields = prm.get_integer ("Number of fields");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Initial temperature model");
      {
        Utilities::AsciiDataBoundary<dim>::parse_parameters(prm);
        prm.enter_subsection ("Litho1.0");
        {
          LAB_isotherm = prm.get_double ("LAB isotherm temperature");
          T0 = prm.get_double ("Surface temperature");
          compensation_depth = prm.get_double ("Temperature compensation depth");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      // Retrieve the indices of the fields that represent the lithospheric layers.
      AssertThrow(this->introspection().compositional_name_exists("upper"),ExcMessage("We need a compositional field called 'upper' representing the upper crust."));
      AssertThrow(this->introspection().compositional_name_exists("lower"),ExcMessage("We need a compositional field called 'lower' representing the lower crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_L"),ExcMessage("We need a compositional field called 'mantle_L' representing the lithospheric part of the mantle."));

      // For now, we assume a 3-layer system with an upper crust, lower crust and lithospheric mantle
      const unsigned int id_upper = this->introspection().compositional_index_for_name("upper");
      const unsigned int id_lower = this->introspection().compositional_index_for_name("lower");
      const unsigned int id_mantle_L = this->introspection().compositional_index_for_name("mantle_L");

      // Retrieve other material properties set in different sections such that there
      // is no need to set them twice.

      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Compositional heating");
        {
          // The heating model compositional heating prefixes an entry for the background material
          const std::vector<double> temp_heat_productivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Compositional heating values"))),
                                                               n_fields+1,
                                                               "Compositional heating values");
          // This sets the heat productivity in W/m3 units
          heat_productivities.push_back(temp_heat_productivities[id_upper+1]);
          heat_productivities.push_back(temp_heat_productivities[id_lower+1]);
          heat_productivities.push_back(temp_heat_productivities[id_mantle_L+1]);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Visco Plastic");
        {
          // The material model viscoplastic prefixes an entry for the background material
          const std::vector<double> temp_densities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Densities"))),
                                                     n_fields+1,
                                                     "Densities");
          const std::vector<double> temp_thermal_diffusivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal diffusivities"))),
                                                                 n_fields+1,
                                                                 "Thermal diffusivities");
          const std::vector<double> temp_heat_capacities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Heat capacities"))),
                                                           n_fields+1,
                                                           "Heat capacities");

          densities.push_back(temp_densities[id_upper+1]);
          densities.push_back(temp_densities[id_lower+1]);
          densities.push_back(temp_densities[id_mantle_L+1]);

          // Thermal diffusivity kappa = k/(rho*cp), so thermal conducitivity k = kappa*rho*cp
          conductivities.push_back(temp_thermal_diffusivities[id_upper+1] * densities[0] * temp_heat_capacities[id_upper+1]);
          conductivities.push_back(temp_thermal_diffusivities[id_lower+1] * densities[0] * temp_heat_capacities[id_lower+1]);
          conductivities.push_back(temp_thermal_diffusivities[id_mantle_L+1] * densities[0] * temp_heat_capacities[id_mantle_L+1]);

          // To obtain the radioactive heating rate in W/kg, we divide the volumetric heating rate by density
          AssertThrow(heat_productivities.size() == 3 && densities.size() == 3 && conductivities.size() == 3,
                      ExcMessage("The entries for density, conductivity and heat production do not match with the expected number of layers (3)."))

          for (unsigned int i = 0; i<3; ++i)
            heat_productivities[i] /= densities[i];
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("LITHO1.0");
        {
          upper_crust_fraction = prm.get_double ("Upper crust fraction");
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
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(Litho1,
                                              "litho1",
                                              "Implementation of a model in which the initial "
                                              "temperature is derived from files containing data "
                                              "in ascii format that specify the depth of the Moho and LAB. "
                                              "If a point lies above the LAB depth, it is given a continental geotherm, "
                                              "computed as a steady-state temperature profile including radioactive "
                                              "heating as set in the 'compositional_heating' plugin. "
                                              "Below the LAB, the LAB isotherm is prescribed up to the compensation "
                                              "depht. Below that an adiabatic temperature profile, "
                                              "as retrieved from the plugin set for the adiabatic conditions. "
                                              "The user has to make sure the temperatures at the compensation depth "
                                              "(i.e. the LAB isotherm and the adiabatic temperature) match. "
                                              "Note the required format of the "
                                              "input data: The first lines may contain any number of comments "
                                              "if they begin with '#', but one of these lines needs to "
                                              "contain the number of grid points in each dimension as "
                                              "for example '# POINTS: 3 3'. "
                                              "The order of the data columns "
                                              "has to be 'x', 'Moho depth [m]', 'LAB depth [m]' in a 2d model and "
                                              " 'x', 'y', 'Moho depth [m]', 'LAB depth [m]' in a 3d model. "
                                              "Note that the data in the input "
                                              "files need to be sorted in a specific order: "
                                              "the first coordinate needs to ascend first, "
                                              "followed by the second in order to "
                                              "assign the correct data to the prescribed coordinates. "
                                              "If you use a spherical model, "
                                              "then the data will still be handled as Cartesian, "
                                              "however the assumed grid changes. 'x' will be replaced by "
                                              "the azimuth angle and 'z' by the polar angle measured "
                                              "positive from the north pole. The grid will be assumed to be "
                                              "a latitude-longitude grid. Note that the order "
                                              "of spherical coordinates is 'phi', 'theta' "
                                              "and not 'theta', 'phi', since this allows "
                                              "for dimension independent expressions. "
                                              "When applying temperature boundary conditions, please use "
                                              "the initial temperature plugin.")
  }
}
