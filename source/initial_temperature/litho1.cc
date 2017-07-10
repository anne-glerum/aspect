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
#include <aspect/initial_temperature/litho1.h>

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
      // Find the boundary indicator that represents the surface
      surface_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");

      std::set<types::boundary_id> surface_boundary_set;
      surface_boundary_set.insert(surface_boundary_id);

      // The input ascii table contains two components, the crust depth and the LAB depth
      Utilities::AsciiDataBoundary<dim>::initialize(surface_boundary_set,
                                                    2);
    }


    template <int dim>
    double
    Litho1<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      // We want to get at the LAB depth, which is the second component
      const double LAB_depth = Utilities::AsciiDataBoundary<dim>::get_data_component(surface_boundary_id,
                                                                                     position,
                                                                                     1);

      const double depth = this->get_geometry_model().depth(position);

      if (depth < LAB_depth)
         return (LAB_isotherm - T0)/(LAB_depth) * depth + T0;
      else if (depth < compensation_depth)
         return LAB_isotherm;
      else
         return (1700.0 - LAB_isotherm)/(this->get_geometry_model().maximal_depth()-compensation_depth) * (depth - compensation_depth) + LAB_isotherm; 
    }


    template <int dim>
    void
    Litho1<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Initial temperature model");
      {
        Utilities::AsciiDataBase<dim>::declare_parameters(prm,
                                                          "$ASPECT_SOURCE_DIR/data/initial-temperature/ascii-data/test/",
                                                          "box_2d.txt");
        prm.enter_subsection ("Litho1.0");
        {
          prm.declare_entry ("LAB isotherm temperature", "1673.15",
                             Patterns::Double (0),
                             "The value of the isothermal boundary temperature assumed at the LAB "
                             "and up to the reference depth . Units: Kelvin.");
          prm.declare_entry ("Surface temperature", "273.15",
                             Patterns::Double (0),
                             "The value of the surface temperature. Units: Kelvin.");
          prm.declare_entry ("Bottom temperature", "1700.15",
                             Patterns::Double (0),
                             "The value of the temperature at the bottom of the domain. Units: Kelvin.");
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
      prm.enter_subsection ("Initial temperature model");
      {
        Utilities::AsciiDataBase<dim>::parse_parameters(prm);
        prm.enter_subsection ("Litho1.0");
        {
          LAB_isotherm = prm.get_double ("LAB isotherm temperature");
          T0 = prm.get_double ("Surface temperature");
          T1 = prm.get_double ("Bottom temperature");
          compensation_depth = prm.get_double ("Temperature compensation depth");
        }
        prm.leave_subsection();
/*        prm.enter_subsection ("Litho1.0");
        {
        }
        prm.leave_subsection();
*/      }
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
                                              "otherwise an adiabatic temperature profile. "
                                              " Note the required format of the "
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
