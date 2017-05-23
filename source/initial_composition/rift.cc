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


#include <aspect/initial_composition/rift.h>
#include <aspect/postprocess/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/material_model/visco_plastic.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    Rift<dim>::Rift ()
    {}

    template <int dim>
    void
    Rift<dim>::initialize ()
    {
      AssertThrow(dynamic_cast<const MaterialModel::ViscoPlastic<dim> *>(&this->get_material_model()) != NULL,
                  ExcMessage("This initial condition only makes sense in combination with the visco_plastic material model."));

      // From shear_bands.cc
      Point<dim> extents;
      TableIndices<dim> size_idx;
      for (unsigned int d=0; d<dim; ++d)
        size_idx[d] = grid_intervals[d]+1;

      Table<dim,double> white_noise;
      white_noise.TableBase<dim,double>::reinit(size_idx);
      std_cxx1x::array<std::pair<double,double>,dim> grid_extents;

      if (dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL)
        {
          const GeometryModel::Box<dim> *
          geometry_model
            = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model());

          extents = geometry_model->get_extents();
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("This initial condition only works with the box geometry model."));
        }

      for (unsigned int d=0; d<dim; ++d)
        {
          grid_extents[d].first=0;
          grid_extents[d].second=extents[d];
        }

      // use a fixed number as seed for random generator
      // this is important if we run the code on more than 1 processor
      std::srand(0);

      TableIndices<dim> idx;

      for (unsigned int i=0; i<white_noise.size()[0]; ++i)
        {
          idx[0] = i;
          for (unsigned int j=0; j<white_noise.size()[1]; ++j)
            {
              idx[1] = j;
              if (dim == 3)
                {
                  for (unsigned int k=0; k<white_noise.size()[dim-1]; ++k)
                    {
                      idx[dim-1] = k;
                      // std::rand will give a value between zero and RAND_MAX (usually INT_MAX).
                      // The modulus of this value and 10000, gives a value between 0 and 10000-1.
                      // Subsequently dividing by 5000.0 wil give value between 0 and 2 (excluding 2).
                      // Subtracting 1 will give a range [-1,1)
                      // Because we want values [0,1), we change our white noise computation to:
                      white_noise(idx) = ((std::rand() % 10000) / 10000.0);
                      std::cout << white_noise(idx) << std::endl;
                    }
                }
              else
                white_noise(idx) = ((std::rand() % 10000) / 10000.0);

            }
        }

      interpolate_noise = new Functions::InterpolatedUniformGridData<dim> (grid_extents,
                                                                           grid_intervals,
                                                                           white_noise);
    }
    template <int dim>
    double
    Rift<dim>::
    initial_composition (const Point<dim> &position, const unsigned int n_comp) const
    {
      // For a box it is easy, just drop last coordinate
      const Point<2> surface_position = Point<2>(position[0],position[1]);

      const double distance_to_rift_axis = (dim == 2) ? (position[0]-point_list[0][0]) : std::abs(Utilities::signed_distance_to_polygon<dim>(point_list, surface_position));

      const double depth_smoothing = 0.5 * (1.0 - std::tanh((this->get_geometry_model().depth(position) - strain_depth) / sigma));

      const double noise_amplitude = A * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)))) * depth_smoothing;

      if (n_comp == 0)
        return noise_amplitude * interpolate_noise->value(position);
      else
        return 0.0;
    }

    template <int dim>
    void
    Rift<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Rift");
        {
          prm.declare_entry ("Standard deviation of Gaussian noise amplitude distribution", "20000",
                             Patterns::Double (0),
                             "The standard deviation of the Gaussian distribution of the amplitude of the strain noise. "
                             "Units: $m$.");
          prm.declare_entry ("Maximum amplitude of Gaussian noise amplitude distribution", "0.2",
                             Patterns::Double (0),
                             "The amplitude of the Gaussian distribution of the amplitude of the strain noise. "
                             "Units: $m$.");
          prm.declare_entry ("Depth around which Gaussian noise is smoothed out", "40000",
                             Patterns::Double (0),
                             "The depth around which smoothing out of the strain noise starts with a hyperbolic tangent. "
                             "Units: $m$.");
          prm.declare_entry ("Grid intervals for noise X", "25",
                             Patterns::Integer (0),
                             "Grid intervals in X directions for the white noise added to "
                             "the initial background porosity that will then be interpolated "
                             "to the model grid. "
                             "Units: none.");
          prm.declare_entry ("Grid intervals for noise Y", "25",
                             Patterns::Integer (0),
                             "Grid intervals in Y directions for the white noise added to "
                             "the initial background porosity that will then be interpolated "
                             "to the model grid. "
                             "Units: none.");
          prm.declare_entry ("Grid intervals for noise Z", "25",
                             Patterns::Integer (0),
                             "Grid intervals in Z directions for the white noise added to "
                             "the initial background porosity that will then be interpolated "
                             "to the model grid. "
                             "Units: none.");
          prm.declare_entry("Rift axis polygon",
                            "",
                            Patterns::Anything(),
                            "Set the polygon that represents the rift axis. The polygon is made up of "
                            "a list of points that represent horizontal coordinates (x,y). "
                            "The exact format for the point list describing the polygon is "
                            "\"x1,y1;x2,y2\". The units of the coordinates are "
                            "dependent on the geometry model. In the box model they are in meters, in the "
                            "chunks they are in degrees, etc. Please refer to the manual of the individual "
                            "geometry model to so see how the topography is implemented.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    Rift<dim>::parse_parameters (ParameterHandler &prm)
    {

      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Rift");
        sigma                = prm.get_double ("Standard deviation of Gaussian noise amplitude distribution");
        A                    = prm.get_double ("Maximum amplitude of Gaussian noise amplitude distribution");
        strain_depth         = prm.get_double ("Depth around which Gaussian noise is smoothed out");
        grid_intervals[0]    = prm.get_integer ("Grid intervals for noise X");
        grid_intervals[1]    = prm.get_integer ("Grid intervals for noise Y");
        if (dim == 3)
          grid_intervals[2]    = prm.get_integer ("Grid intervals for noise Z");

        // Read in the polygon string
        const std::string temp_polygon = prm.get("Rift axis polygon");
        // Split the string into point strings
        const std::vector<std::string> temp_coordinates = Utilities::split_string_list(temp_polygon,';');
        const unsigned int n_temp_coordinates = temp_coordinates.size();
        point_list.resize(n_temp_coordinates);
        for (unsigned int i_coord = 0; i_coord < n_temp_coordinates; i_coord++)
          {
            const std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_coordinates[i_coord],','));
            Assert(temp_point.size() == 2,ExcMessage ("The given coordinate '" + temp_coordinates[i_coord] + "' is not correct. "
                                                      "It should only contain 2 parts: "
                                                      "the two coordinates of the polygon point, separated by a ','."));

            point_list[i_coord] = Point<2>(temp_point[0], temp_point[1]);
          }

        if (dim == 3)
          AssertThrow(point_list.size() >= 3, ExcMessage("A polygon should consist of at least 3 points."));
        if (dim == 2)
          AssertThrow(point_list.size() == 1, ExcMessage("In 2D, only one point is needed to specify the rift axis position. "));

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
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(Rift,
                                              "rift",
                                              "Specify the composition in terms of an explicit formula. The format of these "
                                              "functions follows the syntax understood by the "
                                              "muparser library, see Section~\\ref{sec:muparser-format}.")
  }
}
