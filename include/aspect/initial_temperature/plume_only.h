/*
  Copyright (C) 2012 by the authors of the ASPECT code.

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


#ifndef _aspect_initial_temperature_plume_only_h
#define _aspect_initial_temperature_plume_only_h

#include <aspect/initial_temperature/interface.h>
#include <aspect/boundary_temperature/plume_only.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace InitialTemperature
  {
    using namespace dealii;

    /**
     * A class that implements adiabatic initial conditions for the
     * temperature field and, optional, upper and lower thermal boundary
     * layers calculated using the half-space cooling model. The age/depth of
     * the boundary layers are input parameters or are read from a file.
     *
     * @ingroup InitialTemperatures
     */
    template <int dim>
    class PlumeOnly : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
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
        void
        initialize ();

        /**
         * Return the initial temperature as a function of position.
         */
        virtual
        double initial_temperature (const Point<dim> &position) const;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
        /**
         * Pointer to an object that reads and processes data we get from
         * gplates files.
         */
        std::shared_ptr<BoundaryTemperature::internal::PlumeOnlyLookup<dim> > plume_lookup;

        /**
         * Directory in which the plume file is present.
         */
        std::string plume_data_directory;

        /**
         * Filename of plume file.
         */
        std::string plume_file_name;

        /**
           * Magnitude of the temperature anomaly
           */
        double tail_amplitude;

        /**
         * Radius of the temperature anomaly
         */
        double tail_radius;

        /**
         * Magnitude of the plume head temperature anomaly
         */
        double head_amplitude;

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

        /**
         * Whether or not the model domain is a cartesian box.
         */
        bool cartesian = false;

        /**
         * For cartesian models that are offset from the origin,
         * this represents the z-value of the bottom boundary.
         * For spherical domains, this is the radius of the bottom
         * boundary.
         */
        double inner_radius;
        double outer_radius;
    };
  }
}


#endif
