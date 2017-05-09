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
              white_noise(idx) = ((std::rand() % 10000) / 5000.0 - 1.0);
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
      const double noise_amplitude = A * std::exp((-std::pow(position[0]-x_mean,2)/(2.0*std::pow(sigma,2))));

      if (n_comp == 0)
        return noise_amplitude * interpolate_noise->value(position);
      else 
        return function->value(position,n_comp);
    }

    template <int dim>
    void
    Rift<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Rift");
        {
          prm.declare_entry ("Horizontal coordinate of Gaussian mean", "200000",
                             Patterns::Double (0),
                             "The x coordinate of the Gaussian mean of the amplitude of the strain noise. "
                             "Units: $m$.");
          prm.declare_entry ("Standard deviation of Gaussian noise amplitude distribution", "20000",
                             Patterns::Double (0),
                             "The standard deviation of the Gaussian distribution of the amplitude of the strain noise. "
                             "Units: $m$.");
          prm.declare_entry ("Maximum amplitude of Gaussian noise amplitude distribution", "0.2",
                             Patterns::Double (0),
                             "The amplitude of the Gaussian distribution of the amplitude of the strain noise. "
                             "Units: $m$.");
          prm.declare_entry ("Grid intervals for noise X", "100",
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
          Functions::ParsedFunction<dim>::declare_parameters (prm, 1);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    Rift<dim>::parse_parameters (ParameterHandler &prm)
    {
      // we need to get at the number of compositional fields here to
      // initialize the function parser. unfortunately, we can't get it
      // via SimulatorAccess from the simulator itself because at the
      // current point the SimulatorAccess hasn't been initialized
      // yet. so get it from the parameter file directly.
      prm.enter_subsection ("Compositional fields");
      const unsigned int n_compositional_fields = prm.get_integer ("Number of fields");
      prm.leave_subsection ();

      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Rift");
          x_mean              = prm.get_double ("Horizontal coordinate of Gaussian mean");
          sigma              = prm.get_double ("Standard deviation of Gaussian noise amplitude distribution");
          A              = prm.get_double ("Maximum amplitude of Gaussian noise amplitude distribution");
          grid_intervals[0]    = prm.get_integer ("Grid intervals for noise X");
          grid_intervals[1]    = prm.get_integer ("Grid intervals for noise Y");
        try
          {
            function.reset (new Functions::ParsedFunction<dim>(n_compositional_fields));
            function->parse_parameters (prm);
          }
        catch (...)
          {
            std::cerr << "ERROR: FunctionParser failed to parse\n"
                      << "\t'Initial composition model.Function'\n"
                      << "with expression\n"
                      << "\t'" << prm.get("Function expression") << "'\n"
                      << "More information about the cause of the parse error \n"
                      << "is shown below.\n";
            throw;
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
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(Rift,
                                              "rift",
                                              "Specify the composition in terms of an explicit formula. The format of these "
                                              "functions follows the syntax understood by the "
                                              "muparser library, see Section~\\ref{sec:muparser-format}.")
  }
}
