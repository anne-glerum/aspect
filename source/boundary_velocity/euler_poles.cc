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
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/global.h>
#include <aspect/boundary_velocity/euler_poles.h>
#include <aspect/utilities.h>

#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/ellipsoidal_chunk.h>


namespace aspect
{
  namespace BoundaryVelocity
  {

    template <int dim>
    EulerPoles<dim>::EulerPoles ()
    {}


    template <int dim>
    void
    EulerPoles<dim>::initialize ()
    {
      // Check that we're using a 3D spherical model
      AssertThrow (dim == 3,
                ExcMessage ("To use the Euler pole boundary velocity plugin, the model "
                            "must be 3D."));

      AssertThrow (((dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model())) != 0)
          || ((dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model())) != 0)
          || ((dynamic_cast<const GeometryModel::EllipsoidalChunk<dim>*> (&this->get_geometry_model())) != 0),
              ExcMessage ("This Euler pole plugin can only be used when using "
              "a spherical shell or (ellipsoidal) chunk geometry."));

    }


    template <int dim>
    Tensor<1,dim>
    EulerPoles<dim>::
    boundary_velocity (const types::boundary_id boundary_indicator,
                       const Point<dim> &position) const
    {
      // Check if pole was provided for this boundary indicator
      const typename std::map<types::boundary_id, Point<dim> >::const_iterator it = boundary_velocities.find(boundary_indicator);
      AssertThrow (it != boundary_velocities.end(),
                   ExcMessage("You did not provide an Euler pole rotation for this boundary <" + Utilities::int_to_string(boundary_indicator) + ">. "));

      // Compute depth and transition scale factor to transition from outflow to inflow
      const double depth = this->get_geometry_model().depth(position);
      const double transition_scale_factor = std::min(1.,std::max(-1.,(depth - transition_depth) * (-1./transition_width)));

      // Compute the cartesian velocity as the cross product of the pole and the point
      Tensor <1,dim> euler_velocity = cross_product_3d(it->second, position);
      // Scale the velocity for the transition and any user-defined scaling
      euler_velocity *= (transition_scale_factor * scale_factor);

      return euler_velocity;
    }


    template <int dim>
    void
    EulerPoles<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Boundary velocity model");
      {
        prm.enter_subsection ("EulerPoles model");
        {
          prm.declare_entry ("Scale factor", "1",
                             Patterns::Double (0),
                             "Scalar factor, which is applied to the boundary velocity. "
                             "You might want to use this to scale the velocities.");
          prm.declare_entry ("Velocity transition depth", "120000",
                             Patterns::Double (0),
                             "Determines the depth at which the outflow of lithospheric material changes to inflow "
                             "of (asthenospheric) material.");
          prm.declare_entry ("Velocity transition width", "10000",
                             Patterns::Double (0),
                             "Determines the half-width with which the outflow of lithospheric material changes to inflow "
                             "of (asthenospheric) material. A linear transition from outflow to inflow is used. ");
          prm.declare_entry ("Boundary indicator to velocity mappings", "",
                             Patterns::Map (Patterns::Anything(),
                                            Patterns::Double()),
                             "A comma separated list of mappings between boundary "
                             "indicators and the euler pole associated with the "
                             "boundary indicators. The format for this list is "
                             "``indicator1 : value1x; value1y; value1z, indicator2 : value2x; value2y; value2z, ...'', "
                             "where each indicator is a valid boundary indicator "
                             "(either a number or the symbolic name of a boundary as provided "
                             "by the geometry model) "
                             "and each value is a component of the angular velocity vector of that boundary."
                             "The Euler vector is assumed to be in spherical coordinates: lon [degrees], lat [degrees], rotation rate [degrees/My]." );
          // TODO The cross product will give a smaller velocity with radius, need to account for this with balancing inflow
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    EulerPoles<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary velocity model");
      {
        prm.enter_subsection("EulerPoles model");
        {
          scale_factor          = prm.get_double ("Scale factor");
          transition_depth = prm.get_double ("Velocity transition depth");
          transition_width = prm.get_double ("Velocity transition width");

          // get the list of mappings
           const std::vector<std::string> x_boundary_velocities
             = Utilities::split_string_list(prm.get ("Boundary indicator to velocity mappings"));


           for (std::vector<std::string>::const_iterator it = x_boundary_velocities.begin();
                it != x_boundary_velocities.end(); ++it)
             {
               // each entry has the format (white space is optional):
               // <id> : <value1x; value1y; value1z (might have spaces)>
               const std::vector<std::string> parts = Utilities::split_string_list (*it, ':');

               AssertThrow (parts.size() == 2,
                            ExcMessage (std::string("Invalid entry trying to describe boundary "
                                                    "velocities. Each entry needs to have the form "
                                                    "<boundary_id : value1x; value1y; value1z>, "
                                                    "but there is an entry of the form <") + *it + ">"));

               types::boundary_id boundary_id;
               try
                 {
                   boundary_id
                     = this->get_geometry_model().translate_symbolic_boundary_name_to_id (parts[0]);
                 }
               catch (const std::string &error)
                 {
                   AssertThrow (false, ExcMessage ("While parsing the entry <Boundary velocity model/EulerPoles>, "
                                                   "there was an error. Specifically, "
                                                   "the conversion function complained as follows: "
                                                   + error));
                 }

               AssertThrow (boundary_velocities.find(boundary_id) == boundary_velocities.end(),
                            ExcMessage ("Boundary indicator <" + Utilities::int_to_string(boundary_id) +
                                        "> appears more than once in the list of indicators "
                                        "for velocities boundary conditions."));

               const std::vector<std::string> x_pole = Utilities::split_string_list (parts[1], ';');
               AssertThrow (x_pole.size() == dim,
                            ExcMessage (std::string("Invalid entry trying to describe boundary "
                                                    "poles. Each pole entry needs to have the form "
                                                    "<value1x; value1y; value1z>, "
                                                    "but there is an entry of the form <") + parts[1] + ">"));

               std_cxx11::array<double,dim> pole;
               for (unsigned int d=0; d<dim; ++d)
                 pole[d] = Utilities::string_to_double (x_pole[d]);

               // Transform spherical rotation vector to cartesian
               // First go from degrees to radians
               pole[0] = pole[0] * numbers::PI / 180.;
               // Then go from latitude to colatitude
               pole[1] = 0.5*numbers::PI - pole[1]* numbers::PI / 180.;
               // Then go from My to seconds
               pole[dim] = pole[dim] / (1e6 * year_in_seconds) * numbers::PI / 180.;
               // Then go from spherical to cartesian vector
               const Point<dim> cartesian_pole = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(pole);

               boundary_velocities[boundary_id] = cartesian_pole;
             }
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
  namespace BoundaryVelocity
  {
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(EulerPoles,
                                            "euler poles",
                                            "Implementation of a model in which the boundary "
                                            "velocity is derived from Euler poles that describe "
                                            "plate rotations on a sphere.")
  }
}
