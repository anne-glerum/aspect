/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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


#ifndef _aspect_velocity_boundary_conditions_plume_only_h
#define _aspect_velocity_boundary_conditions_plume_only_h

#include <aspect/boundary_velocity/interface.h>
//#include <aspect/boundary_velocity/function.h>

// Additional lookup classes are within these
#include <aspect/boundary_temperature/plume_only.h>

#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace BoundaryVelocity
  {
    using namespace dealii;

    /**
     * A class that implements prescribed velocity boundary conditions
     * determined from a AsciiData input file.
     *
     * @ingroup VelocityBoundaryConditionsModels
     */
    template <int dim>
    class PlumeOnly : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Empty Constructor.
         */
        PlumeOnly ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        virtual
        void
        initialize ();

        /**
         * A function that is called at the beginning of each time step. For
         * the current plugin, this function loads the next data files if
         * necessary and outputs a warning if the end of the set of data
         * files is reached.
         */
        virtual
        void
        update ();

        /**
         * Return the boundary velocity as a function of position. For the
         * current class, this function returns value from the text files.
         */
        Tensor<1,dim>
        boundary_velocity (const types::boundary_id boundary_id,
                           const Point<dim> &position) const;


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
         * Pointer to an object that reads and processes data we get from
         * gplates files.
         */
        std::shared_ptr<BoundaryTemperature::internal::PlumeOnlyLookup<dim> > plume_lookup;

        /**
         * Current plume position. Used to avoid looking up the plume position
         * for every quadrature point, since this is only time-dependent.
         */
        Point<dim> plume_position;

        /**
         * Filename of plume file.
         */
        std::string plume_file_name;

        /**
         * Directory in which the plume file is present.
         */
        std::string plume_data_directory;

        /**
         * Magnitude of the plume velocity anomaly
         */
        double tail_velocity;

        /**
         * Radius of the plume velocity anomaly
         */
        double tail_radius;

        /**
         * Radius of the plume head temperature anomaly
         */
        double head_radius;

        /**
         * Velocity of the plume head inflow
         */
        double head_velocity;

        /**
         * Model time at which the plume tail will start to move according to
         * the position data file. This is equivalent to the difference between
         * model start time (in the past) and the first data time of the plume
         * positions file.
         */
        double model_time_to_start_plume_tail;

        types::boundary_id boundary_id;

        Tensor<1,dim>
        cartesian_velocity(const Point<dim> position) const;

        Tensor<1,dim>
        spherical_velocity(const Point<dim> position) const;

        double
        integrate_plume_inflow ();

//        std::shared_ptr<BoundaryVelocity::Interface<dim> > velocity_function;

        /**
         * The volume of plume material, i.e. the integral of the plume velocity
         * over the bottom boundary of the domain at a specific timestep.
         */
        double volume;

        /**
         * The area of the bottom boundary
         */
        double area;

        /**
         * The distance of the plume head to the boundary.
         * Calculated based on time passed and head velocity.
         */
        double distance_head_to_boundary;

        /**
         * The radius of the plume head in the plane of the bottom boundary.
         */
        double current_head_radius;

        /**
         * The z or R value of the bottom boundary
         */
        double inner_radius;
        double outer_radius;

        /**
         * Whether or not the model domain is a box or spherical
         */
        bool cartesian;

        /**
         * Whether or not to apply the plume inflow velocity normal
         * to the bottom boundary or parallel to the direction of the
         * center of the plume head.
         */
        bool boundary_normal_plume_inflow;
    };
  }
}


#endif
