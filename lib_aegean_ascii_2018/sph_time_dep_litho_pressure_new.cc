/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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


#include "sph_time_dep_litho_pressure_new.h"
#include <aspect/global.h>
#include <aspect/utilities.h>
#include <array>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <aspect/gravity_model/interface.h>
#include <aspect/boundary_composition/interface.h>
#include <aspect/boundary_temperature/interface.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/initial_temperature/interface.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>
//#include "../lib_chunk/chunk_1.h"
//#include <aspect/geometry_model/chunk_4.h>
#include "layered_chunk_4.h"
#include <aspect/geometry_model/ellipsoidal_chunk.h>
#include "two_merged_chunks.h"

namespace aspect
{
  namespace BoundaryTraction
  {
    namespace internal
    {
      template <int dim>
      ComputePressure<dim>::ComputePressure (const std::vector<unsigned int> n_x,
                                             const unsigned int n_z,
                                             const std::set<types::boundary_id> bi,
                                             const bool calculate_initial_p,
                                             const SimulatorAccess<dim> *pointer)
        :
        n_x_entries(n_x),
        n_z_entries(n_z),
        traction_bi(bi),
        n_bi(traction_bi.size()),
        sim_pointer(pointer),
        values(n_bi),
        local_values(n_bi),
        pressure_data(n_bi),
        delta_x(n_bi,0.0),
        delta_z(0.0),
        max_depth(0.0),
        x_range(n_bi,0.0),
        surface_pressure(0.0),
        var_dim(n_bi,1),
        face_id(n_bi,1),
        calc_p(calculate_initial_p)
      {
        // select the lateral binning directions,
        // set the coordinates that are constant along a boundary
        // and set lateral and depth ranges
        compute_fixed_coordinates();

        // initialize the bin sizes
        for (unsigned int n=0; n < n_bi; ++n)
          delta_x[n] = x_range[n] / n_x_entries[n];
        delta_z = max_depth / n_z_entries;

        // set up the GridData input
        interval_endpoints.resize(n_bi);
        n_subintervals.resize(n_bi);
        for (unsigned int n=0; n<n_bi; n++)
          {
            // in 2d there are only vertical intervals
            // and the second statement will override the first.
            n_subintervals[n][dim-2] = n_x_entries[n]-1;
            n_subintervals[n][0] = n_z_entries-1;

            std::pair <double, double> x_endpoints = std::make_pair(fixed_coordinates[n][var_dim[n]]+0.5*delta_x[n],fixed_coordinates[n][var_dim[n]]+x_range[n]-0.5*delta_x[n]);
            std::pair <double, double> z_endpoints = std::make_pair(0.5*delta_z,max_depth-0.5*delta_z);
            interval_endpoints[n][dim-2] = x_endpoints;
            interval_endpoints[n][0] = z_endpoints;
          }


        if (!calc_p)
          sim_pointer->get_pcout() << "    No need for initial pressure profile" << std::endl;
        else
          sim_pointer->get_pcout() << "    Calculating initial pressure profiles for " << n_bi << " traction boundaries." << std::endl;

        // retrieve the surface pressure
        surface_pressure = sim_pointer->get_surface_pressure();

        // TODO:!!! This order does not take into account the fast direction (last index)!!!
        // loop over all table entries to retrieve initial temperature
        // and composition to integrate the pressure downward.
        std::set<types::boundary_id>::const_iterator it = traction_bi.begin();
        for (unsigned int b=0; b<n_bi; ++b, ++it)
          {
            // TODO Only compute the tables if we haven't done so before
            // During initialization, this plugin is called for each traction bi,
            // so we want to avoid computing all tables each time.
            //
            // set up a table with an entry for each bin.
            // the entry will be the average of density, gravity and pressure for each bin.
            // we take the center of the bins as the positions where we now calculate the first pressure.
            TableIndices<dim-1> indices, table_points;
            table_points[dim-2] = n_x_entries[b];
            table_points[0] = n_z_entries;
            // table that only stores the calculated pressure
            Table<dim-1,double> data;
            data.reinit(table_points);
            // tables that store density, gravity, JxW and calculated pressure
            values[b].reinit(table_points);
            local_values[b].reinit(table_points);

            for (unsigned int i=0; i<n_x_entries[b]; ++i)
              {
                indices[dim-2]=i;

                std::array<double,dim> spherical_point;

                double sum = 0;

                if (calc_p)
                  {
                    // set up the input for the density function of the material model
                    typename MaterialModel::Interface<dim>::MaterialModelInputs in0(1, sim_pointer->n_compositional_fields());
                    typename MaterialModel::Interface<dim>::MaterialModelOutputs out0(1, sim_pointer->n_compositional_fields());

                    // set current spherical point
                    // radius is surface radius
                    for (unsigned int d = 0; d<dim; d++)
                      spherical_point[d] = fixed_coordinates[b][d];

                    if (dim == 3)
                      spherical_point[var_dim[b]] = i * delta_x[b] + fixed_coordinates[b][var_dim[b]] + 0.5 * delta_x[b];

                    // convert spherical point to cartesian
                    in0.position[0] = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(spherical_point);

                    // retrieve initial temperature at the top of the domain
                    in0.temperature[0] = sim_pointer->get_initial_temperature_manager().initial_temperature(in0.position[0]);

                    // use surface pressure
                    in0.pressure[0] = surface_pressure;

                    // retrieve initial composition at bin center
                    for (unsigned int c=0; c<sim_pointer->n_compositional_fields(); ++c)
                      in0.composition[0][c] = sim_pointer->get_initial_composition_manager().initial_composition(in0.position[0], c);

                    // we do not need the viscosity to compute density
                    in0.strain_rate.resize(0);

                    // get the density
                    sim_pointer->get_material_model().evaluate(in0, out0);

                    // the second integration point lies at 0.5*delta_z
                    sum = delta_z * 0.5 * 0.5 * out0.densities[0] * sim_pointer->get_gravity_model().gravity_vector(in0.position[0]).norm();

                  }

                for (unsigned int n=0; n<n_z_entries; ++n)
                  {
                    indices[0] = n;

                    AssertThrow(n*delta_z <= max_depth, ExcMessage("Depth associated with data table point larger than max domain depth."));
                    AssertThrow(i*delta_x[b] <= x_range[b], ExcMessage("Horizontal position associated with data table point larger than max domain extent."));

                    // resize the vector elements of the table
                    // to a 4 element vector of zeroes: rho, g, V, P
                    values[b](indices).resize(4,0.0);
                    local_values[b](indices).resize(4,0.0);

                    if (calc_p)
                      {
                        // set up the input for the density function of the material model
                        typename MaterialModel::Interface<dim>::MaterialModelInputs in(1, sim_pointer->n_compositional_fields());
                        typename MaterialModel::Interface<dim>::MaterialModelOutputs out(1, sim_pointer->n_compositional_fields());

                        // use the centers of the bins as position
                        spherical_point[0] = fixed_coordinates[b][0] - (n * delta_z + 0.5 * delta_z);
                        if (dim == 3)
                          spherical_point[var_dim[b]] = i * delta_x[b] + fixed_coordinates[b][var_dim[b]] + 0.5 * delta_x[b];

                        // convert spherical point to cartesian
                        in.position[0] = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(spherical_point);

                        // retrieve initial temperature at bin center
                        in.temperature[0] = sim_pointer->get_initial_temperature_manager().initial_temperature(in.position[0]);

                        // use pressure of bin above or, if needed, surface pressure
                        if (n != 0)
                          indices[0] -= 1;
                        in.pressure[0] = (n != 0) ?
                                         values[b](indices)[3] :
                                         surface_pressure;

                        if (n != 0)
                          indices[0] += 1;

                        // retrieve initial composition at bin center
                        for (unsigned int c=0; c<sim_pointer->n_compositional_fields(); ++c)
                          in.composition[0][c] = sim_pointer->get_initial_composition_manager().initial_composition(in.position[0], c);

                        // we do not need the viscosity to compute density
                        in.strain_rate.resize(0);

                        // get the density
                        sim_pointer->get_material_model().evaluate(in, out);

                        // assign density to table
                        values[b](indices)[0] = out.densities[0];

                        // assign gravity to table
                        values[b](indices)[1] = sim_pointer->get_gravity_model().gravity_vector(in.position[0]).norm();

                        // calculate pressure and assign to table
                        // the distance from the top to the first bin is 0.5*delta_z
                        values[b](indices)[3] = (n != 0) ?
                                                sum + values[b](indices)[0] * values[b](indices)[1] * delta_z * 0.5 :
                                                sum + values[b](indices)[0] * values[b](indices)[1] * delta_z * 0.5 * 0.5;
                        data(indices) = values[b](indices)[3];

                        sum += (n != 0) ?
                               values[b](indices)[0] * values[b](indices)[1] * delta_z :
                               values[b](indices)[0] * values[b](indices)[1] * delta_z * 0.75;

                        Assert(values[b](indices)[3] >= 0.0, ExcInternalError());

                      }
                  }
              }

            if (pressure_data[b])
              delete pressure_data[b];

            pressure_data[b] = new Functions::InterpolatedUniformGridData<dim-1>(interval_endpoints[b],
                                                                                 n_subintervals[b],
                                                                                 data);
          }

      }

      template <int dim>
      void
      ComputePressure<dim>::compute_fixed_coordinates()
      {
        // retrieve origin and extents of the chunk model
        const GeometryModel::Interface<dim> *gm = dynamic_cast<const GeometryModel::Interface<dim>*> (&sim_pointer->get_geometry_model());

        // Chunk
        if (const GeometryModel::Chunk<dim> *chunk = dynamic_cast<const GeometryModel::Chunk<dim> *> (gm))
          {
            origin[0]  = chunk->inner_radius();
            extents[0] = chunk->outer_radius();
            origin[1]  = chunk->west_longitude();
            extents[1] = chunk->east_longitude();
            if (dim == 3)
              {
                origin[2]  = 0.5 * numbers::PI - chunk->north_latitude();
                extents[2] = 0.5 * numbers::PI - chunk->south_latitude();
              }
          }
//        else if (const GeometryModel::Chunk1<dim> *chunk = dynamic_cast<const GeometryModel::Chunk1<dim> *> (gm))
//          {
//            origin[0]  = chunk->inner_radius();
//            extents[0] = chunk->outer_radius();
//            origin[1]  = chunk->west_longitude();
//            extents[1] = chunk->east_longitude();
//            if (dim == 3)
//              {
//                origin[2]  = 0.5 * numbers::PI - chunk->north_latitude();
//                extents[2] = 0.5 * numbers::PI - chunk->south_latitude();
//              }
//          }
//        else if (const GeometryModel::Chunk4<dim> *chunk = dynamic_cast<const GeometryModel::Chunk4<dim> *> (gm))
//          {
//            origin[0]  = chunk->inner_radius();
//            extents[0] = chunk->outer_radius();
//            origin[1]  = chunk->west_longitude();
//            extents[1] = chunk->east_longitude();
//            if (dim == 3)
//              {
//                origin[2]  = 0.5 * numbers::PI - chunk->north_latitude();
//                extents[2] = 0.5 * numbers::PI - chunk->south_latitude();
//              }
//          }
        else if (const GeometryModel::LayeredChunk4<dim> *chunk = dynamic_cast<const GeometryModel::LayeredChunk4<dim> *> (gm))
          {
            origin[0]  = chunk->inner_radius();
            extents[0] = chunk->outer_radius();
            origin[1]  = chunk->west_longitude();
            extents[1] = chunk->east_longitude();
            if (dim == 3)
              {
                origin[2]  = 0.5 * numbers::PI - chunk->north_latitude();
                extents[2] = 0.5 * numbers::PI - chunk->south_latitude();
              }
          }
        else if (const GeometryModel::TwoMergedChunks<dim> *chunk = dynamic_cast<const GeometryModel::TwoMergedChunks<dim> *> (gm))
          {
            origin[0]  = chunk->inner_radius();
            extents[0] = chunk->outer_radius();
            origin[1]  = chunk->west_longitude();
            extents[1] = chunk->east_longitude();
            if (dim == 3)
              {
                origin[2]  = 0.5 * numbers::PI - chunk->north_latitude();
                extents[2] = 0.5 * numbers::PI - chunk->south_latitude();
              }
          }
        // Spherical shell
        else if (const GeometryModel::SphericalShell<dim> *shell = dynamic_cast<const GeometryModel::SphericalShell<dim> *> (gm))
          {
            origin[0]  = shell->inner_radius();
            extents[0] = shell->outer_radius();
            origin[1] = 0.0;
            extents[1] = shell->opening_angle()/180.0*numbers::PI;
            if (dim == 3)
              {
                origin[2] = 0.0;
                if (shell->opening_angle() == 90.0)
                  extents[2] = 0.5 * numbers::PI;
                else
                  extents[2] = numbers::PI;
              }
          }
        else
          AssertThrow(false, ExcNotImplemented());

        // set the maximal z value
        max_depth   = gm->maximal_depth();

        // for each traction boundary, we compute the fixed coordinates
        // and determine which direction is tabled
        fixed_coordinates.resize(n_bi, extents);
        typename std::vector<Point<dim> >::iterator vector_it = fixed_coordinates.begin();

        // iterate over all traction boundary indicators
        // and set the fixed lateral coordinate
        // radius remains zero
        // In 2D the variable dimension is the radius (dim 0)
        // In 3d, the longitudinal or latitudinal direction
        std::set<types::boundary_id>::const_iterator it = traction_bi.begin();
        unsigned int b = 0;
        for (; it!= traction_bi.end(); ++it, ++vector_it, ++b)
          {
            // West boundary
            if (*it == 2 || *it == 6)
              {
                (*vector_it)[1] = origin[1];
                var_dim[b] = 0;
                face_id[b] = 2;
                if (dim == 3)
                  {
                    (*vector_it)[1] = origin[1];
                    (*vector_it)[2] = origin[2];
                    // in this case x stands for the latitudinal binning direction
                    x_range[b] = extents[2] - origin[2];
                    var_dim[b] = 2;
                    sim_pointer->get_pcout() << "    Western boundary " << std::endl;
                  }
              }
            // East boundary
            else if (*it == 3 || *it == 7)
              {
                (*vector_it)[1] = extents[1];
                var_dim[b] = 0;
                face_id[b] = 3;
                if (dim == 3)
                  {
                    (*vector_it)[1] = extents[1];
                    (*vector_it)[2] = origin[2];
                    x_range[b] = extents[2] - origin[2];
                    var_dim[b] = 2;
                    sim_pointer->get_pcout() << "    Eastern boundary " << std::endl;
                  }

              }
            // front boundary (3D)
            else if (dim == 3 && (*it == 4 || *it == 8))
              {
                (*vector_it)[1] = origin[1];
                (*vector_it)[2] = extents[2];
                x_range[b] = extents[1] - origin[1];
                var_dim[b] = 1;
                face_id[b] = 4;
                sim_pointer->get_pcout() << "    Southern boundary " << std::endl;
              }
            // back boundary (3D)
            else if (dim == 3 && (*it == 5 || *it == 9))
              {
                (*vector_it)[1] = origin[1];
                (*vector_it)[2] = origin[2];
                x_range[b] = extents[1] - origin[1];
                var_dim[b] = 1;
                face_id[b] = 5;
                sim_pointer->get_pcout() << "    Northern boundary " << std::endl;
              }
            // top and bottom boundary not implemented
            else
              AssertThrow(false, ExcNotImplemented());


          }

        b = 0;
        it = traction_bi.begin();
        for (; it!= traction_bi.end(); ++it, ++b)
          {
            if (dim==3)
              {
                sim_pointer->get_pcout() << "    Calculated lateral ranges for boundary " << static_cast<int>(*it) << ": Min x: " << origin[var_dim[b]] << " Max x: " << extents[var_dim[b]] << " Range x: " << x_range[b] << std::endl;
                sim_pointer->get_pcout() << "    Fixed coordinates for boundary " << static_cast<int>(*it) << ": " << fixed_coordinates[b][0] << ", " << fixed_coordinates[b][1]*180.0/numbers::PI << ", " << fixed_coordinates[b][dim-1]*180.0/numbers::PI << std::endl;
              }
            else
              {
                sim_pointer->get_pcout() << "    Calculated depth ranges for boundary " << static_cast<int>(*it) << ": Min z: " << origin[var_dim[b]] << " Max z: " << extents[var_dim[b]] << std::endl;
                sim_pointer->get_pcout() << "    Fixed coordinates for boundary " << static_cast<int>(*it) << ": " << fixed_coordinates[b][0] << ", " << fixed_coordinates[b][1]*180.0/numbers::PI << std::endl;
              }

          }


      }

      template <int dim>
      void
      ComputePressure<dim>::extract_pressure(const unsigned n_mid_points)
      {
        sim_pointer->get_pcout() << "    Calculating new pressure profile from previous solution" << std::endl;

        // loop over all cells along the prescribed traction boundary
        // and retrieve density, gravity and volume per quadrature point
        // into bins

        // this quadrature rule yields 20^dim quadrature points evenly distributed in the interior of the cell.
        // We avoid points on the faces, as they would be counted more than once.
        const QIterated<dim> quadrature_formula (QMidpoint<1>(),
                                                 n_mid_points);

        const unsigned int n_q_points = quadrature_formula.size();

        FEValues<dim> fe_values (sim_pointer->get_mapping(),
                                 sim_pointer->get_fe(),
                                 quadrature_formula,
                                 update_values | update_quadrature_points | update_JxW_values);

        const unsigned int n_compositional_fields = sim_pointer->n_compositional_fields();

        std::vector<std::vector<double> > composition_values (n_compositional_fields,std::vector<double> (n_q_points));

        typename DoFHandler<dim>::active_cell_iterator
        cell = sim_pointer->get_dof_handler().begin_active(),
        endc = sim_pointer->get_dof_handler().end();

        MaterialModel::MaterialModelInputs<dim> in(n_q_points,
                                                   n_compositional_fields);
        MaterialModel::MaterialModelOutputs<dim> out(n_q_points,
                                                     n_compositional_fields);

        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              for (unsigned int b = 0; b<n_bi; ++b)
                {
                  TableIndices<dim-1> indices;
                  // TODO: don't recompute for 2 it_bi on the same vertical boundary
                  // There can be two different bi at the same face,
                  // so only look at those cells with the face_id of the bi
                  // TODO: !!! check number face
                  if (cell->face(face_id[b])->at_boundary())
                    {
                      fe_values.reinit (cell);

                      // extract material model inputs.
                      // we need temperature, pressure, position and composition for the density.
                      fe_values[sim_pointer->introspection().extractors.pressure].get_function_values (sim_pointer->get_solution(),
                          in.pressure);
                      fe_values[sim_pointer->introspection().extractors.temperature].get_function_values (sim_pointer->get_solution(),
                          in.temperature);
                      for (unsigned int c=0; c<sim_pointer->n_compositional_fields(); ++c)
                        fe_values[sim_pointer->introspection().extractors.compositional_fields[c]].get_function_values(sim_pointer->get_solution(),
                            composition_values[c]);
                      for (unsigned int i=0; i<n_q_points; ++i)
                        {
                          in.position[i] = fe_values.quadrature_point(i);
                          for (unsigned int c=0; c<n_compositional_fields; ++c)
                            in.composition[i][c] = composition_values[c][i];
                        }
                      // we don't need the strain rate
                      in.strain_rate.resize(0);
                      in.cell = &cell;

                      // retrieve material model output
                      sim_pointer->get_material_model().evaluate(in, out);

                      for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                          // convert quadrature point position to depth
                          const double depth = sim_pointer->get_geometry_model().depth(fe_values.quadrature_point(q));
                          // calculate into which vertical bin this point falls
                          const unsigned int id_z = static_cast<unsigned int>((depth*n_z_entries)/max_depth);
                          //AssertThrow(id_z<n_z_entries, ExcInternalError());
                          AssertThrow(id_z<n_z_entries, ExcMessage("z bin number "
                                                                   + dealii::Utilities::int_to_string(id_z)
                                                                   + " larger than number of z entries "
                                                                   + dealii::Utilities::int_to_string(n_z_entries)
                                                                   + " for depth "
                                                                   + dealii::Utilities::int_to_string(static_cast<int>(depth))
                                                                   + " and max depth "
                                                                   + dealii::Utilities::int_to_string(static_cast<int>(max_depth))));
                          AssertThrow(id_z>=0, ExcMessage("z bin number smaller than zero."));
                          AssertThrow(id_z < local_values[b].size(0), ExcMessage("z bin index larger than table size."));

                          // in 3D we need to know which lateral direction is to be tabled
                          // we need to rescale the longitude to the -pi,pi interval
                          std::array<double,dim> spherical_quad_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(fe_values.quadrature_point(q));
                          if (spherical_quad_point[1] > numbers::PI)
                            spherical_quad_point[1] -= 2.0*numbers::PI;
                          const double x = spherical_quad_point[var_dim[b]] - fixed_coordinates[b][var_dim[b]];

                          // calculate into which horizontal bin this point falls
                          // in 2D it is always bin 0
                          const unsigned int id_x = (dim == 3) ?
                                                    static_cast<unsigned int>((x*n_x_entries[b])/x_range[b]) :
                                                    0;
                          AssertThrow(id_x<n_x_entries[b], ExcMessage("Lateral bin higher than max bin."));
                          AssertThrow(id_x>=0, ExcMessage("Lateral bin lower than min bin."));
                          if (dim == 3)
                            AssertThrow(id_x < local_values[b].size(1), ExcMessage("x bin index larger than table size."));

                          AssertThrow(out.densities[q] < 6000.0 && out.densities[q] >= 1.0, ExcMessage("Density too high."));

                          indices[dim-2] = id_x;
                          indices[0] = id_z;

                          // assign the retrieved values
                          // density
                          local_values[b](indices)[0] += out.densities[q] * fe_values.JxW(q);

                          //gravity
                          local_values[b](indices)[1] += sim_pointer->get_gravity_model().gravity_vector(fe_values.quadrature_point(q)).norm() * fe_values.JxW(q);

                          // volume
                          local_values[b](indices)[2] += fe_values.JxW(q);
                        }
                    }
                }
            }

        sim_pointer->get_pcout() << "    Extracted gravity and density values " << std::endl;

        // TODO can we do one MPI sum, for example if we roll out the table to a vector?

        // TODO: this must be costly.
        for (unsigned int b=0; b<n_bi; ++b)
          {
            TableIndices<dim-1> indices;
            for (unsigned int i=0; i<n_x_entries[b]; ++i)
              {
                for (unsigned int n=0; n<n_z_entries; ++n)
                  {
                    indices[dim-2] = i;
                    indices[0] = n;

                    // sum the local vector components and place them in "values", replacing old vectors
                    dealii::Utilities::MPI::sum(local_values[b](indices),sim_pointer->get_mpi_communicator(),values[b](indices));

                    // divide by volume of cells.
                    // if bins were not hit,
                    // prescribe the previous values.
                    if (values[b](indices)[2] > 0.0)
                      {
                        values[b](indices)[0]=values[b](indices)[0]/values[b](indices)[2];
                        values[b](indices)[1]=values[b](indices)[1]/values[b](indices)[2];
                      }
                    else
                      {
                        // this happens when the number of bins gets too high for a certain resolution
                        // One can play with the number of q_points in each cell or increase resolution
                        // or decrease the number of bins
                        sim_pointer->get_pcout() << "No hits in bin with (z,x): " << n << "," << i << std::endl;
                        // if n == 0, there is no previous value
                        TableIndices<dim-1> previous_indices = indices, next_indices = indices;
                        previous_indices[0] -= 1;
                        next_indices[0] += 1;

                        if (n != 0)
                          {
                            values[b](indices)[0] = values[b](previous_indices)[0];
                            values[b](indices)[1] = values[b](previous_indices)[1];
                          }
                        else
                          {
                            values[b](indices)[0] = values[b](next_indices)[0];
                            values[b](indices)[1] = values[b](next_indices)[1];
                          }
                      }

                    // post-condition
                    Assert(values[b](indices)[0] >= 1.0 && values[b](indices)[0]<6000.0, ExcMessage("Density too high: "
                                                                                                    + dealii::Utilities::int_to_string(static_cast<int> (values[b](indices)[0]))
                                                                                                    + " column: "
                                                                                                    + dealii::Utilities::int_to_string(i)
                                                                                                    + " row: "
                                                                                                    + dealii::Utilities::int_to_string(n)));
                    Assert(values[b](indices)[1] >= 0.0 && values[b](indices)[1]<100.0, ExcMessage("Gravity magnitude too high."));

                    // set the local values to 0 again for next update
                    local_values[b](indices).assign(4,0.0);
                  }
              }
          }

        sim_pointer->get_pcout() << "   Updated bin values " << std::endl;

        // now integrate pressure downward using the trapezoid rule
        for (unsigned int b=0; b<n_bi; ++b)
          {
            TableIndices<dim-1> indices, table_points;
            table_points[dim-2] = n_x_entries[b];
            table_points[0] = n_z_entries;
            Table<dim-1,double> data;
            data.reinit(table_points);
            for (unsigned int i=0; i<n_x_entries[b]; ++i)
              {
                // set up the input for the density function of the material model
                typename MaterialModel::Interface<dim>::MaterialModelInputs in0(1, sim_pointer->n_compositional_fields());
                typename MaterialModel::Interface<dim>::MaterialModelOutputs out0(1, sim_pointer->n_compositional_fields());

                // get the density and gravity at the top of the domain
                std::array<double,dim> spherical_point;
                for (unsigned int d = 0; d<dim; d++)
                  spherical_point[d] = fixed_coordinates[b][d];

                if (dim == 3)
                  spherical_point[var_dim[b]] = i * delta_x[b] + fixed_coordinates[b][var_dim[b]] + 0.5 * delta_x[b];

                // convert spherical point to cartesian
                in0.position[0] = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(spherical_point);

                // retrieve temperature at top of domain
                in0.temperature[0] = sim_pointer->get_boundary_temperature().boundary_temperature(1, in0.position[0]);

                // use surface pressure
                in0.pressure[0] = surface_pressure;

                // retrieve initial composition at top of domain
                for (unsigned int c=0; c<sim_pointer->n_compositional_fields(); ++c)
                  in0.composition[0][c] = sim_pointer->get_boundary_composition().boundary_composition(1, in0.position[0], c);

                // we do not need the viscosity to compute density
                in0.strain_rate.resize(0);

                // get the density
                sim_pointer->get_material_model().evaluate(in0, out0);

                double sum = delta_z * 0.5 * 0.5 * out0.densities[0] * sim_pointer->get_gravity_model().gravity_vector(in0.position[0]).norm();

                for (unsigned int n=0; n<n_z_entries; ++n)
                  {
                    indices[dim-2] = i;
                    indices[0] = n;

                    // if n==0, integration distance is 0.5*delta_z
                    values[b](indices)[3] = (n != 0) ?
                                            sum + values[b](indices)[0] * values[b](indices)[1] * delta_z * 0.5 :
                                            sum + values[b](indices)[0] * values[b](indices)[1] * delta_z * 0.5 * 0.5;

                    data(indices) =  values[b](indices)[3];
                    sim_pointer->get_pcout() << indices << " " << data(indices) << std::endl;

                    sum += (n != 0) ?
                           values[b](indices)[0] * values[b](indices)[1] * delta_z :
                           values[b](indices)[0] * values[b](indices)[1] * delta_z * 0.75;

                  }
              }

            if (pressure_data[b])
              delete pressure_data[b];

            pressure_data[b] = new Functions::InterpolatedUniformGridData<dim-1>(interval_endpoints[b],
                                                                                 n_subintervals[b],
                                                                                 data);
          }

        sim_pointer->get_pcout() << "   Computed pressure " << std::endl;
      }


      template <int dim>
      double
      ComputePressure<dim>::get_pressure(const Point<dim-1> point, const unsigned int bi) const
      {
        // pre-condition
        AssertThrow(bi < pressure_data.size(), ExcMessage("No pressure data for boundary "
                                                          + dealii::Utilities::int_to_string(bi)));
        AssertThrow(point[0] > interval_endpoints[bi][0].first-delta_z, ExcMessage("Point falls outside table."));
        AssertThrow(point[0] < interval_endpoints[bi][0].second+delta_z, ExcMessage("Point falls outside table."));
        if (dim == 3)
          {
            AssertThrow(point[1] > interval_endpoints[bi][1].first-delta_x[bi], ExcMessage("Point falls outside table."));
            AssertThrow(point[1] < interval_endpoints[bi][1].second+delta_x[bi], ExcMessage("Point falls outside table."));
          }

        return pressure_data[bi]->value(point);
      }


      template <int dim>
      std::vector<double>
      ComputePressure<dim>::get_delta_x() const
      {
        return delta_x;
      }

      template <int dim>
      double
      ComputePressure<dim>::get_delta_z() const
      {
        Assert(delta_z > 0.0, ExcInternalError());
        return delta_z;
      }

      template <int dim>
      std::vector< unsigned int>
      ComputePressure<dim>::get_tabulated_dimension() const
      {
        return var_dim;
      }

      template <int dim>
      Point<dim>
      ComputePressure<dim>::get_origin() const
      {
        return origin;
      }

      template <int dim>
      Point<dim>
      ComputePressure<dim>::get_extents() const
      {
        return extents;
      }

    }




    template <int dim>
    STDLP<dim>::STDLP ()
      :
      n_x_bins(1,0),
      n_z_bins(0),
      n_mid_points(10),
      var_dim(1,0)
    {}


    template <int dim>
    void
    STDLP<dim>::initialize()
    {
      // This plugin is only for spherical, coordinate parallel geometries
      AssertThrow (dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model()) != 0 ||
                   dynamic_cast<const GeometryModel::TwoMergedChunks<dim>*> (&this->get_geometry_model()) != 0 ||
//                   dynamic_cast<const GeometryModel::Chunk1<dim>*> (&this->get_geometry_model()) != 0 ||
//                   dynamic_cast<const GeometryModel::Chunk4<dim>*> (&this->get_geometry_model()) != 0 ||
                   dynamic_cast<const GeometryModel::LayeredChunk4<dim>*> (&this->get_geometry_model()) != 0 ||
                   dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()) != 0,
                   ExcMessage("This boundary traction plugin can only be used for chunk or spherical shell geometries."));

      // Ensure the initial lithostatic pressure traction boundary conditions are used,
      // and register for which boundary indicators these conditions are set.
      // Also construct a mapping between the boundary indicator and the internal numbering
      // of traction boundaries used here.
      // TODO get_traction_boundary_conditions does not return all bi before the loop over all
      // traction boundaries is completed in core.cc
      const std::map<types::boundary_id,std::shared_ptr<BoundaryTraction::Interface<dim> > >
      tbc = this->get_boundary_traction();

      std::set<types::boundary_id> temp_traction_bi;
      for (typename std::map<types::boundary_id,std::shared_ptr<BoundaryTraction::Interface<dim> > >::const_iterator
           p = tbc.begin();
           p != tbc.end(); ++p)
        if (p->second.get() == this)
          temp_traction_bi.insert(p->first);

      AssertThrow(*(temp_traction_bi.begin()) != numbers::invalid_boundary_id,
                  ExcMessage("Did not find any boundary indicators for the initial lithostatic pressure plugin."));

      traction_bi = this->get_traction_boundary_indicators();
      std::set<types::boundary_id>::const_iterator it_bi = traction_bi.begin();
      unsigned int b = 0;
      for (; it_bi != traction_bi.end(); ++it_bi, ++b)
        {
          bi_map[*it_bi] = b;
          this->get_pcout() << "   Initializing boundary traction " << static_cast<int> (*it_bi) << std::endl;
          AssertThrow(*it_bi != 0 && *it_bi != 1, ExcMessage("Top and bottom boundary cannot use this boundary traction plugin."));

        }

      AssertThrow(n_x_bins.size() == traction_bi.size(), ExcMessage("The number of boundaries for which a horizontal bin number is specified "
                                                                    "does not correspond to the number of traction boundaries."));

      // initialize the compute pressure class
      compute_pressure.reset(new internal::ComputePressure<dim>(n_x_bins,n_z_bins,traction_bi,calculate_initial_p,this));

      // get the lateral dimension that is tabulated in 3D cases
      var_dim = compute_pressure->get_tabulated_dimension();

      // output some info
      it_bi = traction_bi.begin();
      b = 0;
      for (; it_bi != traction_bi.end(); ++it_bi, ++b)
        {
          this->get_pcout() << "   Pressure grid for boundary indicator " << static_cast<unsigned int>(*it_bi) << std::endl;
          this->get_pcout() << "   " << n_x_bins[b] << " x " << n_z_bins << " bins are used, resulting in a spacing of " <<  std::endl;
          if (dim==3)
            this->get_pcout() << "   " << compute_pressure->get_delta_x()[b]*180.0/numbers::PI  << " [degrees] x " << std::endl;
          this->get_pcout() << "   " << compute_pressure->get_delta_z() << " [m]" << std::endl;
          if (dim == 3)
            this->get_pcout() << "   Tabulated direction: " << compute_pressure->get_tabulated_dimension()[b] << std::endl;
          this->get_pcout() << std::endl;
        }
    }

    template <int dim>
    void
    STDLP<dim>::
    update()
    {
      // For t0 we compute the lithostatic pressure based on the
      // initial composition and temperature conditions.
      // At the beginning of subsequent timesteps,
      // we retrieve the density from the previous solution and
      // thus compute the lithostatic pressure.
      if ((!calculate_initial_p && this->get_pre_refinement_step() == amr) || this->get_timestep_number() > 0)
        {
          this->get_pcout() << "   Updating pressure grids" << std::endl;
          compute_pressure->extract_pressure(n_mid_points);
        }
    }


    template <int dim>
    Tensor<1,dim>
    STDLP<dim>::
    boundary_traction (const types::boundary_id bi,
                       const Point<dim> &p,
                       const Tensor<1,dim> &normal) const
    {
      // Find the internal number of the given boundary indicator
      const std::map<types::boundary_id, unsigned int>::const_iterator b = bi_map.find(bi);
      // Assert we have an internal number for this indicator
      AssertThrow(b != bi_map.end(), ExcMessage("This boundary indicator is not found."));

      // We want to set the component normal to the vertical boundary
      // to the lithostatic pressure, the rest of the traction
      // components are set to zero.
      // We get the lithostatic pressure from a (bi)linear interpolation of
      // the calculated profile for the boundary with the given number.
      Tensor<1,dim> traction;
      traction = -get_pressure(p, b->second) * normal;
      return traction;
    }

    template <int dim>
    double
    STDLP<dim>::
    get_pressure (const Point<dim> &p,
                  const unsigned int bi) const
    {
      // convert point to spherical coordinates
      // radius, longitude (, latitude)
      std::array<double,dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(p);
      // correct longitude from 0,2PI to -PI,PI interval if domain crosses 0 meridian
      const Point<dim> domain_extents = compute_pressure->get_extents();
      const Point<dim> domain_origin = compute_pressure->get_origin();
      // TODO: This might give problem
      if (spherical_point[1] > numbers::PI && domain_origin[1] < 0.0 && domain_extents[1] > 0.0 && spherical_point[1] > domain_extents[1])
        spherical_point[1] -= 2.0*numbers::PI;

      // construct the boundary point
      // depth (and lon/lat)
      Point<dim-1> boundary_point;
      boundary_point[dim-2] = spherical_point[var_dim[bi]];
      boundary_point[0] = this->get_geometry_model().depth(p);

      return compute_pressure->get_pressure (boundary_point, bi);
    }


    template <int dim>
    void
    STDLP<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Time and position dependent lithostatic pressure");
        {
          prm.declare_entry ("Number of lateral bins", "100",
                             Patterns::List(Patterns::Integer(0)),
                             "The number of columns into which the open boundary face is divided. "
                             "Should be > 2*n_cells(x). Unit:/. "
                             "This number is ignored in 2D computations. ");
          prm.declare_entry ("Number of radial bins", "100",
                             Patterns::Integer(0),
                             "The number of rows into which the open boundary face is divided. "
                             "Should be > 2*n_cells(z). Unit:/. ");
          prm.declare_entry ("Number of midpoint quadrature points base", "10",
                             Patterns::Integer(0),
                             "The base N of the N^dim number of quadrature points that are inserted in each "
                             "boundary cell. Unit:/. ");

          prm.declare_entry ("Calculate pressure grids from initial conditions", "false",
                             Patterns::Bool(),
                             "If true, the initial conditions are used to calculate the initial pressure "
                             "grids, otherwise they are interpolated from the finite element solution.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    STDLP<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Time and position dependent lithostatic pressure");
        {
          // TODO specify with boundary names. e.g. east: 200
          // read list of numbers of bins
          const std::vector<int >n_x =
            dealii::Utilities::string_to_int(dealii::Utilities::split_string_list(prm.get("Number of lateral bins")));

          const unsigned int n_bi = n_x.size();
          n_x_bins.resize(n_bi);
          // in 2D there will be only 1 profile along the vertical boundary
          if (dim == 2)
            n_x_bins.assign(n_bi,1);
          else
            {
              // assign bin numbers
              for (unsigned int i = 0; i < n_bi; ++i)
                n_x_bins[i]  = static_cast<unsigned int>(n_x[i]);
            }

          n_z_bins = prm.get_integer("Number of radial bins");

          n_mid_points = prm.get_integer("Number of midpoint quadrature points base");

          calculate_initial_p = prm.get_bool("Calculate pressure grids from initial conditions");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();


      unsigned int refinement;
      prm.enter_subsection("Mesh refinement");
      {
        amr        = prm.get_integer("Initial adaptive refinement");
        refinement = amr + prm.get_integer("Initial global refinement");
      }
      prm.leave_subsection();

      // Check whether n_*_bins is denser than twice the number of cells in * direction
//      if (dim == 3)
//        for (unsigned int b=0; b<n_x_bins.size(); ++b)
//          AssertThrow(2.0*std::pow(2.0,refinement) <= n_x_bins[b], ExcMessage("Not enough lateral bins for this resolution. Refinement level: "
//                                                                              + dealii::Utilities::int_to_string(refinement)
//                                                                              + ", and number of lateral bins: "
//                                                                              + dealii::Utilities::int_to_string(n_x_bins[b])));

      AssertThrow(std::pow(2.0,refinement) <= n_z_bins, ExcMessage("Not enough radial bins for this resolution."));
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(STDLP,
                                                 "spherical time and position dependent lithostatic pressure",
                                                 "Implementation of a model in which the boundary "
                                                 "traction is given in terms of a normal traction component "
                                                 "set to the lithostatic pressure "
                                                 "calculated according to the parameters in section "
                                                 "``Boundary traction model|Lithostatic pressure''. "
                                                 "\n\n"
                                                 "The lithostatic pressure is calculated by integrating "
                                                 "the pressure downward based on the initial composition "
                                                 "and temperature. "
                                                 "\n\n"
                                                 "Note that the tangential velocity component(s) should be set "
                                                 "to zero. ")
  }
}
