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


#ifndef _aspect_boundary_velocity_euler_poles_h
#define _aspect_boundary_velocity_euler_poles_h

#include <aspect/boundary_velocity/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/std_cxx11/array.h>


namespace aspect
{
  namespace BoundaryVelocity
  {
    using namespace dealii;

    /**
     * A class that implements prescribed velocity boundary conditions
     * determined from angular rotation vectors specified for each boundary
     * with prescribed velocity boundary conditions.
     *
     * @ingroup BoundaryVelocities
     */
    template <int dim>
    class EulerPoles : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Empty Constructor.
         */
        EulerPoles ();

        /**
         * Return the boundary velocity as a function of position and boundary
         * indicator. For the current class, this function returns the value
         * of the cross product of the angular velocity vector and the point vector.
         */
        Tensor<1,dim>
        boundary_velocity (const types::boundary_id boundary_indicator,
                           const Point<dim> &position) const;

        // avoid -Woverloaded-virtual warning until the deprecated function
        // is removed from the interface:
        using Interface<dim>::boundary_velocity;

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        virtual
        void
        initialize ();

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm);

      private:
        /**
         * Scale the velocity boundary condition by a scalar factor.
         */
        double scale_factor;
        double area_scale_factor;
        double transition_area_scale_factor;

        /**
         * The outer and inner radius of the model domain.
         */
        double outer_radius;
        double inner_radius;

        /**
         * The depth of the transition from outflow to inflow. This transition
         * occurs linearly around the transition depth, where the gradient is determined
         * by the transition width. This transition ensures that the integrated
         * prescribed flow through the boundaries is zero.
         */
        double transition_depth;

        /**
         * The half width of the linear transition zone.
         */
        double transition_width;

        /**
         * Parameters describing the transition in terms of radius.
         */
        double transition_radius;
        double transition_radius_max;
        double transition_radius_min;

        /**
         * A map of spherical rotation vectors for each boundary indicator.
         */
        std::map<types::boundary_id, std::vector<Point<dim> > > boundary_velocities;
        std::map<types::boundary_id, std::vector<double> > boundary_transitions;
    };
  }
}


#endif
