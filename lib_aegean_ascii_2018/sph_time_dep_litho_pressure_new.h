/*
  Copyright (C) 2011, 2012 by the authors of the ASPECT code.

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


#ifndef __aspect__traction_boundary_conditions_sph_time_dep_litho_pressure_h
#define __aspect__traction_boundary_conditions_sph_time_dep_litho_pressure_h

#include <aspect/boundary_traction/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/function_lib.h>


namespace aspect
{
  namespace BoundaryTraction
  {
    using namespace dealii;


    namespace internal
    {
      template <int dim>
      class ComputePressure
      {
        public:
          ComputePressure (const std::vector<unsigned int> n_x,
                           const unsigned int n_z,
                           const std::set<types::boundary_id> bi,
                           const bool calculate_initial_p,
                           const SimulatorAccess<dim> *pointer);

          /**
           * Returns the lithostatic pressure at the given point
           * based on the calculated pressure GridData for each traction boundary.
           */
          double get_pressure(const Point<dim-1> boundary_point,
                              const unsigned int bi) const;

          /**
           * Recomputes pressure GridData based on the solution of the
           * previous timestep.
           */
          void extract_pressure(const unsigned n_mid_points);

          /**
           * Returns the lateral spacing of the pressure bins
           * for each traction boundary.
           */
          std::vector<double> get_delta_x() const;

          /**
           * Returns the radial spacing of the pressure bins.
           */
          double get_delta_z() const;

          /**
           * Returns the lateral dimension that is tabulated for
           * each traction boundary in 3D.
           */
          std::vector<unsigned int> get_tabulated_dimension () const;

          /**
           * Returns the origin of the domain in terms of minimum
           * radius, longitude and colatitude.
           */
          Point<dim> get_origin() const;

          /**
           * Returns the extent of the domain in terms of maximum
           * radius, longitude and colatitude.
           */
          Point<dim> get_extents() const;

        private:

          /**
           * Computes the coordinate that is constant on each traction
           * boundary in 3D and sets the tabulated dimensions.
           */
          void compute_fixed_coordinates();


          /**
           * The number of lateral table entries per traction boundary.
           */
          std::vector<unsigned int> n_x_entries;

          /**
           * The number of radial table entries.
           */
          unsigned int n_z_entries;

          /**
           * The boundary ids of boundaries with prescribed tractions
           * calculated by this plugin.
           */
          std::set<types::boundary_id> traction_bi;

          /*
           * The number of traction boundaries
           */
          unsigned int n_bi;

          /*
           * The pointer to the simulator access
           */
          const SimulatorAccess<dim> *sim_pointer;

          /**
           * The table that stores the density, gravity and volume
           * values for each bin.
           * For multiple bi, the vector lists the table for each bi.
           */
          std::vector<Table<dim-1,std::vector<double> > > values;
          std::vector<Table<dim-1,std::vector<double> > > local_values;

          /**
           * The GridData that stores the lithostatic pressures
           * for each traction boundary.
           */
          std::vector< Function<dim-1> *> pressure_data;

          /**
           * The spacing in the x direction per bi.
           */
          std::vector< double> delta_x;

          /**
           * The spacing in the z direction.
           */
          double delta_z;

          /**
           * The maximum depth of the model.
           */
          double max_depth;

          /**
           * The x range of the model.
           */
          std::vector <double> x_range;

          /**
           * The pressure set by the user at the surface of the domain
           */
          double surface_pressure;

          /**
           *
           */
          std::vector< Point<dim> > fixed_coordinates;

          std::vector<std::array< std::pair< double, double >, dim-1 > > interval_endpoints;
          /**
           * The number of intervals in each direction per bi
           */
          std::vector<std::array< unsigned int, dim-1 > > n_subintervals;

          std::vector< unsigned int> var_dim;

          std::vector< unsigned int> face_id;

          /**
           * The lowest and highest values of each coordinate:
           * radius, longitude, colatitude
           */
          Point<dim> origin;
          Point<dim> extents;

          bool calc_p;

      };
    }



    /**
     * A class that implements traction boundary conditions based on a
     * functional description provided in the input file.
     *
     * @ingroup BoundaryTractionModels
     */
    template <int dim>
    class STDLP : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */
        STDLP ();

        /**
         * Initialization function. Because this function is called after
         * initializing the SimulatorAccess, all of the necessary information
         * is available to calculate the pressure profile.
         */
        virtual void initialize ();


        /**
         * Update function.
         */
        void update ();


        /**
         * Return the boundary traction as a function of position. The
         * (outward) normal vector to the domain is also provided as
         * a second argument.
         */
        virtual
        Tensor<1,dim>
        boundary_traction (const types::boundary_id boundary_indicator,
                           const Point<dim> &position,
                           const Tensor<1,dim> &normal_vector) const;


        /**
         * Declare the parameters this class takes through input files. The
         * default implementation of this function does not describe any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         * The default implementation of this function does not read any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:

        std::set<types::boundary_id> traction_bi;


        /**
         * The number of bins in the x direction.
         */
        std::vector<unsigned int> n_x_bins;

        /**
         * The number of bins in the z direction.
         * Only needed in 3D computations.
         */
        unsigned int n_z_bins;

        unsigned int n_mid_points;

        /**
         * Pointer to class
         */
        std::shared_ptr<internal::ComputePressure<dim> > compute_pressure;


        /**
         * Return the lithostatic pressure at a given point of the domain.
         */
        double get_pressure (const Point<dim> &p, const unsigned int boundary_indicator) const;

        /**
         * The tabulated lateral dimension
         */
        std::vector< unsigned int> var_dim;

        std::map<types::boundary_id, unsigned int> bi_map;

        /**
         * The number of adaptive initial refinement cycles
         */
        unsigned int amr;

        /**
         * Whether or not to use the initial conditions to calculate pressure grids
         */
        bool calculate_initial_p;


    };
  }
}


#endif
