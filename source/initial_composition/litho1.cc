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
#include <aspect/initial_composition/litho1.h>


namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    Litho1<dim>::Litho1 ()
    {}


    template <int dim>
    void
    Litho1<dim>::initialize ()
    {
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
    }


    template <int dim>
    double
    Litho1<dim>::
    initial_composition (const Point<dim> &position,
                         const unsigned int n_comp) const
    {
      // The Moho depth is the first component
      const double Moho_depth = Utilities::AsciiDataBoundary<dim>::get_data_component(surface_boundary_id,
                                                                                      position,
                                                                                      0);
      // The LAB depth is the second component
      const double LAB_depth = std::max(Moho_depth, Utilities::AsciiDataBoundary<dim>::get_data_component(bottom_boundary_id,
                                        position,
                                        0));

      AssertThrow(Moho_depth <= LAB_depth, ExcMessage("The crust is thicker than the LAB."));

      // The depth of the point under investigation
      const double depth = this->get_geometry_model().depth(position);

      // The upper crustal field (with field id 0)
      if (depth < Moho_depth*upper_crust_fraction && n_comp == 0)
        return 1.;
      // The lower crustal field (with field id 1 (or 0))
      if (depth >= Moho_depth*upper_crust_fraction && depth < Moho_depth && n_comp == lower_crust_id)
        return 1.;
      // The lithospheric mantle
      else if (depth >= Moho_depth && depth < LAB_depth && n_comp == lower_crust_id+1)
        return 1.;
      // Everything else, which is not represented by a compositional field.
      else
        return 0.;
    }


    template <int dim>
    void
    Litho1<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        Utilities::AsciiDataBoundary<dim>::declare_parameters(prm,
                                                          "$ASPECT_SOURCE_DIR/data/initial-composition/ascii-data/test/",
                                                          "box_2d_%s.%d.txt");
        prm.enter_subsection("LITHO1.0");
        {
          prm.declare_entry ("Upper crust fraction", "0.66",
                             Patterns::Double (0,1),
                             "A number that specifies the fraction of the Moho depth "
                             "that will be used to set the thickness of the upper crust "
                             "instead of reading its thickness from the ascii table. "
                             "For a fraction of 0, no compositional field is set for the "
                             "upper crust. "
                             "Unit: -.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    Litho1<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        Utilities::AsciiDataBoundary<dim>::parse_parameters(prm);
        prm.enter_subsection("LITHO1.0");
        {
          upper_crust_fraction = prm.get_double ("Upper crust fraction");

          // If there is no upper crust, the compositional field number
          // of the lower crust is the same as that of the upper crust (i.e. they are the same field)
          if (upper_crust_fraction == 0 && !this->introspection().compositional_name_exists("upper"))
            lower_crust_id = 0;
          else
            lower_crust_id = 1;
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
  namespace InitialComposition
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(Litho1,
                                              "litho1",
                                              "Implementation of a model in which the initial "
                                              "composition is derived from files containing data "
                                              "about the depth of the Moho and LAB "
                                              "in ascii format. "
                                              "Either one compositional field is used to represent "
                                              "the total crust up to Moho depth, "
                                              "or an upper crust composition is "
                                              "specified with the upper\_crust\_fraction parameter that"
                                              "sets an upper crustal field from 0 to "
                                              "upper\_crust\_fraction*Moho\_depth. "
                                              "Note the required format of the "
                                              "input data: The first lines may contain any number of comments "
                                              "if they begin with '#', but one of these lines needs to "
                                              "contain the number of grid points in each dimension as "
                                              "for example '# POINTS: 3 3'. "
                                              "The order of the data columns "
                                              "has to be 'x', 'Moho depth [m]', 'LAB depth [m]', "
                                              "etc. in a 2d model and 'x', 'y', 'Moho depth [m]', "
                                              "'LAB depth [m]', etc. in a 3d model. "
                                              "Note that the data in the input "
                                              "files need to be sorted in a specific order: "
                                              "the first coordinate needs to ascend first, "
                                              "followed by the second in order to "
                                              "assign the correct data to the prescribed coordinates. "
                                              "If you use a spherical model, "
                                              "then the data will still be handled as Cartesian, "
                                              "however the assumed grid changes. 'x' will be replaced by "
                                              "by the azimuth angle and 'y' by the polar angle measured "
                                              "positive from the north pole. The grid will be assumed to be "
                                              "a latitude-longitude grid. Note that the order "
                                              "of spherical coordinates is 'phi', 'theta' "
                                              "and not 'theta', 'phi', since this allows "
                                              "for dimension independent expressions.")
  }
}
