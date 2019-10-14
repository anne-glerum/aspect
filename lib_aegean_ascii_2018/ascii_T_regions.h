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


#ifndef __aspect__initial_temperature_ascii_T_regions_h
#define __aspect__initial_temperature_ascii_T_regions_h

#include <aspect/initial_temperature/interface.h>
#include <aspect/simulator_access.h>
//#include <array>

namespace aspect
{
  namespace InitialTemperature
  {
    using namespace dealii;

    namespace internal
    {
      template <int dim>
      class SlabGridLookup
      {
        public:
          SlabGridLookup (const unsigned int n_slab,
                          const std::vector<unsigned int> n_hor,
                          const std::vector<unsigned int> n_ver,
                          const std::string &point_file,
                          const ConditionalOStream &pcout);

          Point<dim> get_local_coord_point(const unsigned int slab_nr, const unsigned int i_hor, const unsigned int j_ver) const;

          double get_arc_length(const unsigned int slab_nr, const unsigned int i_hor, const unsigned int j_ver) const;

        private:


//         unsigned int n_grid_coord;
          unsigned int n_slabs;

          std::vector<unsigned int> n_hor_points, n_ver_points;

          std::vector<dealii::Table<2, Point<dim> > > grid_coord;
          std::vector<dealii::Table<2, double> > arc_length;
      };

      template <int dim>
      class PolygonLookup
      {
        public:
          PolygonLookup (const std::vector<unsigned int> n_points_per_polygon,
                         const std::string &polygon_file,
                         const ConditionalOStream &pcout);

          std::vector<Point<dim> > get_cartesian_polygon(const unsigned int polygon_nr) const;

          std::vector<Point<dim> > get_spherical_polygon(const unsigned int polygon_nr) const;

          /*
           * Function that converts Cartesian coordinates to spherical coordinates
           */
          Point <dim> cart_to_spherical(const Point<dim> coord) const;
          /*
           * Function that converts spherical coordinates to Cartesian coordinates
           */
          Point <dim> spherical_to_cart(const Point<dim> spherical_coord) const;

        private:

          unsigned int n_polygons;
          unsigned int n_polygon_points;

          std::vector<std::vector<Point<dim> > > cartesian_polygons;
          std::vector<std::vector<Point<dim> > > spherical_polygons;

      };

    }

    /**
     * A class that describes an initial temperature field for a
     * box geometry model. The temperature is according to the plate cooling model
     * unless the queried point lies in the slab. Then the temperature follows
     * McKenzie 1970.
     *
     * @ingroup InitialConditionsModels
     */

    template <int dim>
    class AsciiPip : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Initialization function. Loads the material data and sets up
         * pointers.
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

        /**
         * A function that returns whether the given point lies inside a certain 2D polygon.
         * True means inside the polygon.
         */
        bool is_inside_2D_polygon(const unsigned int n, const Point<dim> coord) const;


      private:

        /**
         * A function that calculates the temperature according to the plate
         * cooling model
         */
        double oceanic_plate_temperature (const unsigned int field_nr,
                                          const double depth,
                                          const double plate_thickness = 7e4,
                                          const double plate_depth_top = 0.0, //40500.0,
                                          const double plate_age = 6e7 * year_in_seconds) const;

        /**
         * A function that calculates the temperature based on the thickness of the plate
         * and the depth of the point within that plate according to a linear profile
         */
        double continental_plate_temperature (const double depth,
                                              const double plate_thickness = 120e3,
                                              const double plate_depth_top = 0.0 /*38200.0*/) const;

        /**
         * A function that calculates the temperature according to the linear continental
         * profile or the plate cooling oceanic profile or an errorfunction mix of both
         * depending on the distance to the 2D polygon boundary denoting the region
         * of different plate type
         */
        /**
         * Enumeration for selecting between different plate types
         */
        enum CompoType
        {
          mantle,
          air,
          oceanic_plate,
          continental_plate,
          continental_oceanic_plate,
          oceanic_continental_plate,
          continental_blacksea_plate,
          Aegea,
          Nubia,
          Eurasia,
          slab,
          continental_weak_zone,
          mixed_continental_weak_zone,
          oceanic_weak_zone

        };
        double mixed_plate_temperature (const unsigned int field_nr, const CompoType plate_type, const Point<dim> coord) const;

        /**
         * A function that takes a point within the slab and calculates its
         * temperature according to McKenzie 1970
         */
        double slab_temperature (const unsigned int field_nr, const Point<dim> coord) const;

        /**
         * A function that takes a point within the slab and calculates its
         * local arc length and depth within the slab
         */
        Tensor<1,2> slab_length_and_depth (const unsigned int slab_nr, const Point<dim> coord) const;

        /**
         * A function that returns the projection of a point onto a triangular plane
         */
        void point_to_plane(const Point<dim> coord, const std::vector<Point<dim> > plane, double &normal_distance, Point<dim> &point_in_plane, bool &in_triangle) const;

        /**
         * A function that returns the local coordinates of the projected point,
         * ie the coordinates of the triangle, along its edges that parallel the grid
         */
        void triangle_coord(const Point<dim> coord, const std::vector<Point<dim> > triangle, Tensor<1,2> &triangle_coord) const;


        double triangle_basis_function_interpolation(const std::vector<Point<dim> > triangle, const Tensor<1,3> values, const Point<dim> point) const;

        /**
         * A function that returns the signed distance of the given point to a certain 2D polygon.
         * Negative means outside, positive inside the region of the polygon.
         */
        double distance_to_polygon(const unsigned int n, const Point<dim> coord) const;

        /**
         * Pointer to an object that reads and processes the coordinates
         * of the regular slab surface grid.
         */
        std::shared_ptr<internal::SlabGridLookup<dim> > grid_lookup;

        /**
         * Pointer to an object that reads and processes the coordinates
         * of the polygons that represent different regions within plates.
         */
        std::shared_ptr<internal::PolygonLookup<dim> > polygon_lookup;

        /**
         * File directory and names
         */
        std::string datadirectory;
        std::string slab_grid_file_name;
        std::string polygons_file_name;

        /**
         * The number of compositional fields
         */
        unsigned int n_compositional_fields;

        ////////////////////////
        // Volume information //
        ////////////////////////
        /**
         * The number of plates for which a triangulated volume exists.
         */
        unsigned int n_volumes;

        /**
         * The field numbers per volume.
         */
        std::vector <std::vector <unsigned int> > fields_per_volume;

        /**
         * The indicator of the region(s) within each field that represents a mixed plate
         * The length of this vector equals the number of fields
         * But the indicator for fields that are not plates is set to 10 and is not used
         */
        std::vector<std::vector<unsigned int> > region_nr_per_volume;

        /**
         * The indicator of the region(s) within each field that represents a mixed plate
         * The length of this vector equals the number of fields
         * But the indicator for fields that are not plates is set to 10 and is not used
         */
        std::vector<unsigned int> region_nr_per_field;

        /**
         * The number of the slab if a field represents a slab
         * The length of this vector equals the number of fields
         * But the indicator for fields that are not slabs is set to 10 and is not used
         */
        std::vector<unsigned int> slab_nr_per_volume;

        /**
         * The number of polygons that describe a region in a volume
         * either with a separate composition or not
         */ 
        unsigned int n_regions;
 

        //////////////////////
        // Slab information //
        //////////////////////
        /**
         * The number of slab volumes
         */
        unsigned int n_slab_fields;

        /**
         * The maximum depth any of the slabs will reach
         */
        double max_slab_depth;

        /**
         * Number of horizontal points in slab surface grid per slab
         */
        std::vector<unsigned int> n_hor_grid_points;

        /**
         * Number of vertical points in slab surface grid per slab
         */
        std::vector<unsigned int> n_ver_grid_points;

        /**
         * The subduction velocity needed for initial T in slab (McKenzie 1970) per slab
         */
        std::vector<double> subduction_vel;

        /**
         * The slab thickness needed for initial T in slab (McKenzie 1970) per slab
         */
        std::vector<double> slab_thickness;

        ////////////////////////////////
        // Region polygon information //
        ////////////////////////////////
        /**
         * The number of polygon regions
         */
        std::vector<unsigned int> n_points_per_polygon;

        /**
         * The total nr of points of all polygons
         */
        unsigned int n_polygon_points;

        /**
         * Whether or not to incorporate the EEC (it drives significant mantle flow icw OB)
         */
        bool include_EEC;


        /////////////////////////////
        // Temperature information //
        /////////////////////////////
        /**
         * The surface temperature
         */
        double T_top;

        /**
         * The asthenospheric potential temperature
         */
        mutable double T_a;

        /**
         * The temperature at the bottom of the domain
         */
        double T_bottom;

        /**
         * The reference temperature
         */
        double T_0;

        /**
         * Whether or not to adjust the McKenzie temperature in the slab to the overriding plate
         */
        bool compensate_trench_temp;


        ///////////////////////
        // Plate information //
        ///////////////////////


        std::vector<CompoType> composition_types;

        double max_air_thickness;

        double max_lithosphere_depth;

        class Plate
        {
          private:
            double thickness;
            double age;
            double depth_top;
            bool oceanic;

          public:
            Plate()
              : thickness(120000.0), age(8e7), depth_top(0.0), oceanic(false)
            {

            }

            Plate(double plate_thickness, double plate_age, double plate_depth_top, bool oceanic_plate)
              : thickness(plate_thickness), age(plate_age), depth_top(plate_depth_top), oceanic(oceanic_plate)
            {

            }

            double get_thickness() const
            {
              return thickness;
            }

            double get_age() const
            {
              return age;
            }

            double get_depth_top() const
            {
              return depth_top;
            }

            bool is_plate_oceanic() const
            {
              return oceanic;
            }

            void set_thickness(const double plate_thickness)
            {
              thickness = plate_thickness;
            }

            void set_age(const double plate_age)
            {
              age = plate_age;
            }

            void set_depth_top(const double plate_depth_top)
            {
              depth_top = plate_depth_top;
            }

            void set_plate_type(const bool oceanic_plate)
            {
              oceanic = oceanic_plate;
            }



        };

        Plate Oceanic_Plate;
        Plate Continental_Plate;
        Plate Black_Sea_Plate;
        Plate Thin_Continental_Plate;
        Plate Thin_Old_Oceanic_Plate;
        Plate Thick_Old_Oceanic_Plate;
        Plate EEC_Plate;
        Plate Anatolia_Plate;
        Plate Young_Oceanic_Plate;
        Plate Old_Oceanic_Plate;

        std::vector<Plate> plate_type_per_region;
        ////////////////////////////
        // Refinement information //
        ////////////////////////////
        unsigned int initial_global_refinement;
        unsigned int refinement_limit;

        // The radius of the top of the model domain
        double R_model;

        // Pointer to the temperature background plugin
        std::shared_ptr<InitialTemperature::Interface<dim> > ascii_model;
    };

  }
}

#endif
