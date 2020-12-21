/*
 *  ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike.h"

#ifdef HAVE_GSL

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

using namespace nest;

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike >
  mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike >::create()
{
  insert_( names::V_m, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_y_elem_< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::V_M > );
  insert_( names::theta, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_y_elem_< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::THETA > );
  insert_(
    names::g_AMPA, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_y_elem_< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::G_AMPA > );
  insert_( names::g_NMDA, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_g_NMDA_ );
  insert_(
    names::g_GABA_A, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_y_elem_< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::G_GABA_A > );
  insert_(
    names::g_GABA_B, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_y_elem_< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::G_GABA_B > );
  insert_(
      "g_GABA_B1a", &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_g_GABA_B1a );
  insert_(
      "x_GABA_B1a", &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_x_GABA_B1a ); // Returns AMPA
  insert_( names::I_NaP, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_I_NaP_ );
  insert_( names::I_KNa, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_I_KNa_ );
  insert_( names::I_T, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_I_T_ );
  insert_( names::I_h, &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_I_h_ );
  insert_(
      "I_gap", &mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_I_gap_ );
}
}

/* ----------------------------------------------------------------
 * Iteration function
 * ---------------------------------------------------------------- */

extern "C" inline int
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_dynamics( double time,
    const double y[],
    double f[],
    void* pnode )
{
  // shorthand
  typedef mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_ S;

  // get access to node so we can almost work as in a member class
  assert( pnode );
  mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike& node = *( reinterpret_cast< mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike* >( pnode ) );

  // easier access to membrane potential, clamp if requested
  const double& V = node.P_.voltage_clamp ? node.V_.V_clamp_ : y[ S::V_M ];

  /*
   * NMDA conductance
   *
   * We need to take care to handle instantaneous blocking correctly.
   * If the unblock-variables m_{fast,slow}_NMDA are greater than the
   * equilibrium value m_eq_NMDA for the present membrane potential, we cannot
   * change m_NMDA_{fast,slow} values in State_[], since the ODE Solver may
   * call this function multiple times and in arbitrary temporal order. We thus
   * need to use local variables for the values at the current time, and check
   * the state variables once the ODE solver has completed the time step.
   */
  const double m_eq_NMDA = node.m_eq_NMDA_( V );
  const double m_fast_NMDA = std::min( m_eq_NMDA, y[ S::m_fast_NMDA ] );
  const double m_slow_NMDA = std::min( m_eq_NMDA, y[ S::m_slow_NMDA ] );
  const double m_NMDA = node.m_NMDA_( V, m_eq_NMDA, m_fast_NMDA, m_slow_NMDA );

  // Calculate sum of all synaptic channels.
  // Sign convention: For each current, write I = - g * ( V - E )
  //    then dV/dt ~ Sum(I)
  const double I_syn = -y[ S::G_AMPA ] * ( V - node.P_.E_rev_AMPA )
    - y[ S::G_NMDA_TIMECOURSE ] * m_NMDA * ( V - node.P_.E_rev_NMDA )
    - y[ S::G_GABA_A ] * ( V - node.P_.E_rev_GABA_A )
    - y[ S::G_GABA_B ] * ( V - node.P_.E_rev_GABA_B );

  // The post-spike K-current, only while refractory
  const double I_spike =
    node.S_.ref_steps_ > 0 ? -( V - node.P_.E_K ) / node.P_.tau_spike : 0.0;

  // leak currents
  const double I_Na = -node.P_.g_NaL * ( V - node.P_.E_Na );
  const double I_K = -node.P_.g_KL * ( V - node.P_.E_K );

  // intrinsic currents
  // I_Na(p), m_inf^3 according to Compte et al, J Neurophysiol 2003 89:2707
  const double INaP_thresh = -55.7;
  const double INaP_slope = 7.7;
  const double m_inf_NaP =
    1.0 / ( 1.0 + std::exp( -( V - INaP_thresh ) / INaP_slope ) );
  node.S_.I_NaP_ = -node.P_.g_peak_NaP * std::pow( m_inf_NaP, 3.0 )
    * ( V - node.P_.E_rev_NaP );

  // I_DK
  const double d_half = 0.25;
  const double m_inf_KNa =
    1.0 / ( 1.0 + std::pow( d_half / y[ S::D_IKNa ], 3.5 ) );
  node.S_.I_KNa_ = -node.P_.g_peak_KNa * m_inf_KNa * ( V - node.P_.E_rev_KNa );

  // I_T
  node.S_.I_T_ = -node.P_.g_peak_T * y[ S::m_IT ] * y[ S::m_IT ] * y[ S::h_IT ]
    * ( V - node.P_.E_rev_T );

  // I_h
  node.S_.I_h_ = -node.P_.g_peak_h * y[ S::m_Ih ] * ( V - node.P_.E_rev_h );

  // Gap junction

  // set I_gap depending on interpolation order
  double gap = 0.0;

  const double t = time / node.B_.step_;

  switch ( kernel().simulation_manager.get_wfr_interpolation_order() )
  {
  case 0:
    gap = -node.B_.sumj_g_ij_ * V
      + node.B_.interpolation_coefficients[ node.B_.lag_ ];
    break;

  case 1:
    gap = -node.B_.sumj_g_ij_ * V
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 2 + 0 ]
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 2 + 1 ] * t;
    break;

  case 3:
    gap = -node.B_.sumj_g_ij_ * V
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 0 ]
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 1 ] * t
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 2 ] * t * t
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 3 ] * t * t * t;
    break;

  default:
    throw BadProperty( "Interpolation order must be 0, 1, or 3." );
  }

  node.S_.I_gap_ = node.P_.g_gap_scale_factor * gap;


  // delta V
  f[ S::V_M ] =
    ( I_Na + I_K + I_syn + node.S_.I_NaP_ + node.S_.I_KNa_ + node.S_.I_T_
      + node.S_.I_h_ + node.B_.I_stim_ + node.S_.I_gap_ ) / node.P_.tau_m
    + I_spike;

  // delta theta
  f[ S::THETA ] = -( y[ S::THETA ] - node.P_.theta_eq ) / node.P_.tau_theta;

  // Synaptic channels

  // AMPA
  f[ S::DG_AMPA ] = -y[ S::DG_AMPA ] / node.P_.tau_rise_AMPA;
  f[ S::G_AMPA ] = y[ S::DG_AMPA ] - y[ S::G_AMPA ] / node.P_.tau_decay_AMPA;

  // NMDA
  f[ S::DG_NMDA_TIMECOURSE ] =
    -y[ S::DG_NMDA_TIMECOURSE ] / node.P_.tau_rise_NMDA;
  f[ S::G_NMDA_TIMECOURSE ] = y[ S::DG_NMDA_TIMECOURSE ]
    - y[ S::G_NMDA_TIMECOURSE ] / node.P_.tau_decay_NMDA;
  f[ S::m_fast_NMDA ] = ( m_eq_NMDA - m_fast_NMDA ) / node.P_.tau_Mg_fast_NMDA;
  f[ S::m_slow_NMDA ] = ( m_eq_NMDA - m_slow_NMDA ) / node.P_.tau_Mg_slow_NMDA;

  // GABA_A
  f[ S::DG_GABA_A ] = -y[ S::DG_GABA_A ] / node.P_.tau_rise_GABA_A;
  f[ S::G_GABA_A ] =
    y[ S::DG_GABA_A ] - y[ S::G_GABA_A ] / node.P_.tau_decay_GABA_A;

  // GABA_B
  f[ S::DG_GABA_B ] = -y[ S::DG_GABA_B ] / node.P_.tau_rise_GABA_B;
  f[ S::G_GABA_B ] =
    y[ S::DG_GABA_B ] - y[ S::G_GABA_B ] / node.P_.tau_decay_GABA_B;

  // GABA_B1a
  // We model GABA_B1a  as a pre-synaptic removal of excitation. Since we cannot
  // model the GABA_B conductances on the synapses, we model them on the neuron
  // (as difference of exponentials). All the GABA_B spikes onto the neuron are
  // combined in a single time-course. When spikes are transmitted to
  // the neuron, their amplitude is scaled down by an amount depending on the
  // instantaneous GABA_B1a conductance.
  f[ S::DG_GABA_B1a ] = -y[ S::DG_GABA_B1a ] / node.P_.tau_rise_GABA_B1a;
  f[ S::G_GABA_B1a ] =
    y[ S::DG_GABA_B1a ] - y[ S::G_GABA_B1a ] / node.P_.tau_decay_GABA_B1a;

  // I_KNa
  f[ S::D_IKNa ] = ( node.D_eq_KNa_( V ) - y[ S::D_IKNa ] ) / node.P_.tau_D_KNa;

  // I_T
  const double tau_m_T = 0.22
      / ( std::exp( -( V + 132.0 ) / 16.7 ) + std::exp( ( V + 16.8 ) / 18.2 ) )
    + 0.13;
  const double tau_h_T = 8.2
    + ( 56.6 + 0.27 * std::exp( ( V + 115.2 ) / 5.0 ) )
      / ( 1.0 + std::exp( ( V + 86.0 ) / 3.2 ) );
  f[ S::m_IT ] = ( node.m_eq_T_( V ) - y[ S::m_IT ] ) / tau_m_T;
  f[ S::h_IT ] = ( node.h_eq_T_( V ) - y[ S::h_IT ] ) / tau_h_T;

  // I_h
  const double tau_m_h =
    1.0 / ( std::exp( -14.59 - 0.086 * V ) + std::exp( -1.87 + 0.0701 * V ) );
  f[ S::m_Ih ] = ( node.m_eq_h_( V ) - y[ S::m_Ih ] ) / tau_m_h;

  return GSL_SUCCESS;
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::m_eq_h_( double V ) const
{
  const double I_h_Vthreshold = -75.0;
  return 1.0 / ( 1.0 + std::exp( ( V - I_h_Vthreshold ) / 5.5 ) );
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::h_eq_T_( double V ) const
{
  return 1.0 / ( 1.0 + std::exp( ( V + 83.0 ) / 4 ) );
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::m_eq_T_( double V ) const
{
  return 1.0 / ( 1.0 + std::exp( -( V + 59.0 ) / 6.2 ) );
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::D_eq_KNa_( double V ) const
{
  const double D_influx_peak = 0.025;
  const double D_thresh = -10.0;
  const double D_slope = 5.0;
  const double D_eq = 0.001;

  const double D_influx =
    D_influx_peak / ( 1.0 + std::exp( -( V - D_thresh ) / D_slope ) );
  return P_.tau_D_KNa * D_influx + D_eq;
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::m_eq_NMDA_( double V ) const
{
  return 1.0 / ( 1.0 + std::exp( -P_.S_act_NMDA * ( V - P_.V_act_NMDA ) ) );
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::m_NMDA_( double V,
  double m_eq,
  double m_fast,
  double m_slow ) const
{
  const double A1 = 0.51 - 0.0028 * V;
  const double A2 = 1 - A1;
  return P_.instant_unblock_NMDA ? m_eq : A1 * m_fast + A2 * m_slow;
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_g_NMDA_() const
{
  return S_.y_[ State_::G_NMDA_TIMECOURSE ]
    * m_NMDA_( S_.y_[ State_::V_M ],
           m_eq_NMDA_( S_.y_[ State_::V_M ] ),
           S_.y_[ State_::m_fast_NMDA ],
           S_.y_[ State_::m_slow_NMDA ] );
}

inline double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_x_GABA_B1a( int i_receptor ) const {
  return std::max(
    0.0,
    std::min(
      P_.xmax_GABA_B1a,
      1.0 - P_.alpha_GABA_B1a[ i_receptor ] * get_g_GABA_B1a()
    )
  );
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::Parameters_::Parameters_()
  : E_Na( 30.0 ) // mV
  , E_K( -90.0 ) // mV
  , g_NaL( 0.2 )
  , g_KL( 1.0 )
  , tau_m( 16.0 )     // ms
  , theta_eq( -51.0 ) // mV
  , tau_theta( 2.0 )  // ms
  , tau_spike( 1.75 ) // ms
  , t_ref( 2.0 )      // ms
  , g_peak_AMPA( 0.1 )
  , tau_rise_AMPA( 0.5 )  // ms
  , tau_decay_AMPA( 2.4 ) // ms
  , E_rev_AMPA( 0.0 )     // mV
  , g_peak_NMDA( 0.075 )
  , tau_rise_NMDA( 4.0 )     // ms
  , tau_decay_NMDA( 40.0 )   // ms
  , E_rev_NMDA( 0.0 )        // mV
  , V_act_NMDA( -25.57 )     // mV
  , S_act_NMDA( 0.081 )      // mV
  , tau_Mg_slow_NMDA( 22.7 ) // ms
  , tau_Mg_fast_NMDA( 0.68 ) // ms
  , instant_unblock_NMDA( false )
  , g_peak_GABA_A( 0.33 )
  , tau_rise_GABA_A( 1.0 )  // ms
  , tau_decay_GABA_A( 7.0 ) // ms
  , E_rev_GABA_A( -70.0 )   // mV
  , g_peak_GABA_B( 0.0132 )
  , tau_rise_GABA_B( 60.0 )   // ms
  , tau_decay_GABA_B( 200.0 ) // ms
  , E_rev_GABA_B( -90.0 )     // mV
  , xmax_GABA_B1a( 1.0 )   // (1)
  , g_peak_GABA_B1a( 0.0 )     // (nS)
  , g_max_GABA_B1a( 1.0 )
  , tau_rise_GABA_B1a( 60.0 )   // (msec)
  , tau_decay_GABA_B1a( 200.0 ) // (msec)
  , alpha_GABA_B1a( MINI - 1 , 1.0 ) // (nS-1)
  , mini_mean(0.5)    //mV
  , mini_sigma(0.25)  //mV
  , g_gap_scale_factor( 1.0 ) // (unitless)
  , g_peak_NaP( 0.5 )
  , E_rev_NaP( 30.0 ) // mV
  , g_peak_KNa( 0.5 )
  , E_rev_KNa( -90.0 )  // mV
  , tau_D_KNa( 1250.0 ) // ms
  , g_peak_T( 1.0 )
  , E_rev_T( 0.0 ) // mV
  , g_peak_h( 1.0 )
  , E_rev_h( -40.0 ) // mV
  , voltage_clamp( false )
{
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::State_( const ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike& node, const Parameters_& p )
  : ref_steps_( 0 )
  , I_NaP_( 0.0 )
  , I_KNa_( 0.0 )
  , I_T_( 0.0 )
  , I_h_( 0.0 )
{
  // initialize with equilibrium values
  y_[ V_M ] = ( p.g_NaL * p.E_Na + p.g_KL * p.E_K ) / ( p.g_NaL + p.g_KL );
  y_[ THETA ] = p.theta_eq;

  for ( size_t i = 2; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = 0.0;
  }

  y_[ m_fast_NMDA ] = node.m_eq_NMDA_( y_[ V_M ] );
  y_[ m_slow_NMDA ] = node.m_eq_NMDA_( y_[ V_M ] );
  y_[ m_Ih ] = node.m_eq_h_( y_[ V_M ] );
  y_[ D_IKNa ] = node.D_eq_KNa_( y_[ V_M ] );
  y_[ m_IT ] = node.m_eq_T_( y_[ V_M ] );
  y_[ h_IT ] = node.h_eq_T_( y_[ V_M ] );
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::State_( const State_& s )
  : ref_steps_( s.ref_steps_ )
  , I_NaP_( s.I_NaP_ )
  , I_KNa_( s.I_KNa_ )
  , I_T_( s.I_T_ )
  , I_h_( s.I_h_ )
{
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = s.y_[ i ];
  }
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_& mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::operator=( const State_& s )
{
  if ( this == &s )
  {
    return *this;
  }

  ref_steps_ = s.ref_steps_;
  I_NaP_ = s.I_NaP_;
  I_KNa_ = s.I_KNa_;
  I_T_ = s.I_T_;
  I_h_ = s.I_h_;

  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = s.y_[ i ];
  }

  return *this;
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::~State_()
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::E_Na, E_Na );
  def< double >( d, names::E_K, E_K );
  def< double >( d, names::g_NaL, g_NaL );
  def< double >( d, names::g_KL, g_KL );
  def< double >( d, names::tau_m, tau_m );
  def< double >( d, names::theta_eq, theta_eq );
  def< double >( d, names::tau_theta, tau_theta );
  def< double >( d, names::t_ref, t_ref );
  def< double >( d, names::tau_spike, tau_spike );
  def< double >( d, names::g_peak_AMPA, g_peak_AMPA );
  def< double >( d, names::tau_rise_AMPA, tau_rise_AMPA );
  def< double >( d, names::tau_decay_AMPA, tau_decay_AMPA );
  def< double >( d, names::E_rev_AMPA, E_rev_AMPA );
  def< double >( d, names::g_peak_NMDA, g_peak_NMDA );
  def< double >( d, names::tau_rise_NMDA, tau_rise_NMDA );
  def< double >( d, names::tau_decay_NMDA, tau_decay_NMDA );
  def< double >( d, names::E_rev_NMDA, E_rev_NMDA );
  def< double >( d, names::V_act_NMDA, V_act_NMDA );
  def< double >( d, names::S_act_NMDA, S_act_NMDA );
  def< double >( d, names::tau_Mg_slow_NMDA, tau_Mg_slow_NMDA );
  def< double >( d, names::tau_Mg_fast_NMDA, tau_Mg_fast_NMDA );
  def< bool >( d, names::instant_unblock_NMDA, instant_unblock_NMDA );
  def< double >( d, names::g_peak_GABA_A, g_peak_GABA_A );
  def< double >( d, names::tau_rise_GABA_A, tau_rise_GABA_A );
  def< double >( d, names::tau_decay_GABA_A, tau_decay_GABA_A );
  def< double >( d, names::E_rev_GABA_A, E_rev_GABA_A );
  def< double >( d, names::g_peak_GABA_B, g_peak_GABA_B );
  def< double >( d, names::tau_rise_GABA_B, tau_rise_GABA_B );
  def< double >( d, names::tau_decay_GABA_B, tau_decay_GABA_B );
  def< double >( d, names::E_rev_GABA_B, E_rev_GABA_B );
  def< double >( d, "xmax_GABA_B1a", xmax_GABA_B1a );
  def< double >( d, "g_peak_GABA_B1a", g_peak_GABA_B1a );
  def< double >( d, "g_max_GABA_B1a", g_max_GABA_B1a );
  // def< double >( d, "alpha_GABA_B1a", alpha_GABA_B1a );
  def< double >( d, "tau_rise_GABA_B1a", tau_rise_GABA_B1a );
  def< double >( d, "tau_decay_GABA_B1a", tau_decay_GABA_B1a );
  def< double >( d, names::g_peak_NaP, g_peak_NaP );
  def< double >( d, names::E_rev_NaP, E_rev_NaP );
  def< double >( d, names::g_peak_KNa, g_peak_KNa );
  def< double >( d, names::E_rev_KNa, E_rev_KNa );
  def< double >( d, names::tau_D_KNa, tau_D_KNa );
  def< double >( d, names::g_peak_T, g_peak_T );
  def< double >( d, names::E_rev_T, E_rev_T );
  def< double >( d, names::g_peak_h, g_peak_h );
  def< double >( d, names::E_rev_h, E_rev_h );
  def< bool >( d, names::voltage_clamp, voltage_clamp );
  def< double >( d, "mini_mean", mini_mean);
  def< double >( d, "mini_sigma", mini_sigma);
  def< double >( d, "g_gap_scale_factor", g_gap_scale_factor);

  // Return alpha_GABA_B1a vector as array datum
  ArrayDatum alpha_GABA_B1a_ad( alpha_GABA_B1a );
  def< ArrayDatum >( d, "alpha_GABA_B1a", alpha_GABA_B1a_ad );
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::E_Na, E_Na );
  updateValue< double >( d, names::E_K, E_K );
  updateValue< double >( d, names::g_NaL, g_NaL );
  updateValue< double >( d, names::g_KL, g_KL );
  updateValue< double >( d, names::tau_m, tau_m );
  updateValue< double >( d, names::theta_eq, theta_eq );
  updateValue< double >( d, names::tau_theta, tau_theta );
  updateValue< double >( d, names::tau_spike, tau_spike );
  updateValue< double >( d, names::t_ref, t_ref );
  updateValue< double >( d, names::g_peak_AMPA, g_peak_AMPA );
  updateValue< double >( d, names::tau_rise_AMPA, tau_rise_AMPA );
  updateValue< double >( d, names::tau_decay_AMPA, tau_decay_AMPA );
  updateValue< double >( d, names::E_rev_AMPA, E_rev_AMPA );
  updateValue< double >( d, names::g_peak_NMDA, g_peak_NMDA );
  updateValue< double >( d, names::tau_rise_NMDA, tau_rise_NMDA );
  updateValue< double >( d, names::tau_decay_NMDA, tau_decay_NMDA );
  updateValue< double >( d, names::E_rev_NMDA, E_rev_NMDA );
  updateValue< double >( d, names::V_act_NMDA, V_act_NMDA );
  updateValue< double >( d, names::S_act_NMDA, S_act_NMDA );
  updateValue< double >( d, names::tau_Mg_slow_NMDA, tau_Mg_slow_NMDA );
  updateValue< double >( d, names::tau_Mg_fast_NMDA, tau_Mg_fast_NMDA );
  updateValue< bool >( d, names::instant_unblock_NMDA, instant_unblock_NMDA );
  updateValue< double >( d, names::g_peak_GABA_A, g_peak_GABA_A );
  updateValue< double >( d, names::tau_rise_GABA_A, tau_rise_GABA_A );
  updateValue< double >( d, names::tau_decay_GABA_A, tau_decay_GABA_A );
  updateValue< double >( d, names::E_rev_GABA_A, E_rev_GABA_A );
  updateValue< double >( d, names::g_peak_GABA_B, g_peak_GABA_B );
  updateValue< double >( d, names::tau_rise_GABA_B, tau_rise_GABA_B );
  updateValue< double >( d, names::tau_decay_GABA_B, tau_decay_GABA_B );
  updateValue< double >( d, names::E_rev_GABA_B, E_rev_GABA_B );
  updateValue< double >( d, "xmax_GABA_B1a", xmax_GABA_B1a );
  updateValue< double >( d, "g_peak_GABA_B1a", g_peak_GABA_B1a );
  updateValue< double >( d, "g_max_GABA_B1a", g_max_GABA_B1a );
  // updateValue< double >( d, "alpha_GABA_B1a", alpha_GABA_B1a );
  updateValue< double >( d, "tau_rise_GABA_B1a", tau_rise_GABA_B1a );
  updateValue< double >( d, "tau_decay_GABA_B1a", tau_decay_GABA_B1a );
  updateValue< double >( d, names::g_peak_NaP, g_peak_NaP );
  updateValue< double >( d, names::E_rev_NaP, E_rev_NaP );
  updateValue< double >( d, names::g_peak_KNa, g_peak_KNa );
  updateValue< double >( d, names::E_rev_KNa, E_rev_KNa );
  updateValue< double >( d, names::tau_D_KNa, tau_D_KNa );
  updateValue< double >( d, names::g_peak_T, g_peak_T );
  updateValue< double >( d, names::E_rev_T, E_rev_T );
  updateValue< double >( d, names::g_peak_h, g_peak_h );
  updateValue< double >( d, names::E_rev_h, E_rev_h );
  updateValue< bool >( d, names::voltage_clamp, voltage_clamp );
  updateValue< double > ( d, "mini_mean", mini_mean );
  updateValue< double > ( d, "mini_sigma", mini_sigma );
  updateValue< double > ( d, "g_gap_scale_factor", g_gap_scale_factor );

  // update and check alpha_GABA_B1a
  if ( updateValue< std::vector< double > >( d, "alpha_GABA_B1a", alpha_GABA_B1a ) )
  {
      if ( alpha_GABA_B1a.size() != MINI - 1 )
      {
          throw BadProperty(
              "The size of the 'alpha_GABA_B1a' array should be equal to the"
              " number of receptors (excluding MINIs)."
          );
      }
      for ( size_t i = 0; i < alpha_GABA_B1a.size(); ++i )
      {
          if ( alpha_GABA_B1a[ i ] < 0.0 )
          {
            throw BadProperty(
              "All alpha_GABA_B1a scaling factors should be between 0 and 1." );
          }
          if ( alpha_GABA_B1a[ i ] > 1.0 )
          {
            throw BadProperty(
              "All alpha_GABA_B1a scaling factors should be between 0 and 1." );
          }
      }
  }

  if ( g_peak_AMPA < 0 )
  {
    throw BadParameter( "g_peak_AMPA >= 0 required." );
  }
  if ( g_peak_GABA_A < 0 )
  {
    throw BadParameter( "g_peak_GABA_A >= 0 required." );
  }
  if ( g_peak_GABA_B < 0 )
  {
    throw BadParameter( "g_peak_GABA_B >= 0 required." );
  }
  if ( g_peak_GABA_B1a < 0 )
  {
    throw BadParameter( "g_peak_GABA_B1a >= 0 required." );
  }
  if ( g_max_GABA_B1a < 0 )
  {
    throw BadParameter( "g_max_GABA_B1a >= 0 required." );
  }
  if ( g_peak_KNa < 0 )
  {
    throw BadParameter( "g_peak_KNa >= 0 required." );
  }
  if ( S_act_NMDA < 0 )
  {
    throw BadParameter( "S_act_NMDA >= 0 required." );
  }
  if ( g_peak_NMDA < 0 )
  {
    throw BadParameter( "g_peak_NMDA >= 0 required." );
  }
  if ( g_peak_T < 0 )
  {
    throw BadParameter( "g_peak_T >= 0 required." );
  }
  if ( g_peak_h < 0 )
  {
    throw BadParameter( "g_peak_h >= 0 required." );
  }
  if ( g_peak_NaP < 0 )
  {
    throw BadParameter( "g_peak_NaP >= 0 required." );
  }
  if ( g_KL < 0 )
  {
    throw BadParameter( "g_KL >= 0 required." );
  }
  if ( g_NaL < 0 )
  {
    throw BadParameter( "g_NaL >= 0 required." );
  }
  if ( g_gap_scale_factor < 0 )
  {
    throw BadParameter( "g_gap_scale_factor >= 0 required." );
  }

  if ( t_ref < 0 )
  {
    throw BadParameter( "t_ref >= 0 required." );
  }

  if ( tau_rise_AMPA <= 0 )
  {
    throw BadParameter( "tau_rise_AMPA > 0 required." );
  }
  if ( tau_decay_AMPA <= 0 )
  {
    throw BadParameter( "tau_decay_AMPA > 0 required." );
  }
  if ( tau_rise_GABA_A <= 0 )
  {
    throw BadParameter( "tau_rise_GABA_A > 0 required." );
  }
  if ( tau_decay_GABA_A <= 0 )
  {
    throw BadParameter( "tau_decay_GABA_A > 0 required." );
  }
  if ( tau_rise_GABA_B <= 0 )
  {
    throw BadParameter( "tau_rise_GABA_B > 0 required." );
  }
  if ( tau_decay_GABA_B <= 0 )
  {
    throw BadParameter( "tau_decay_GABA_B > 0 required." );
  }
  if ( tau_rise_GABA_B1a <= 0 )
  {
    throw BadParameter( "tau_rise_GABA_B1a > 0 required." );
  }
  if ( tau_decay_GABA_B1a <= 0 )
  {
    throw BadParameter( "tau_decay_GABA_B1a > 0 required." );
  }
  if ( tau_rise_NMDA <= 0 )
  {
    throw BadParameter( "tau_rise_NMDA > 0 required." );
  }
  if ( tau_decay_NMDA <= 0 )
  {
    throw BadParameter( "tau_decay_NMDA > 0 required." );
  }
  if ( tau_Mg_fast_NMDA <= 0 )
  {
    throw BadParameter( "tau_Mg_fast_NMDA > 0 required." );
  }
  if ( tau_Mg_slow_NMDA <= 0 )
  {
    throw BadParameter( "tau_Mg_slow_NMDA > 0 required." );
  }
  if ( tau_spike <= 0 )
  {
    throw BadParameter( "tau_spike > 0 required." );
  }
  if ( tau_theta <= 0 )
  {
    throw BadParameter( "tau_theta > 0 required." );
  }
  if ( tau_m <= 0 )
  {
    throw BadParameter( "tau_m > 0 required." );
  }
  if ( tau_D_KNa <= 0 )
  {
    throw BadParameter( "tau_D_KNa > 0 required." );
  }

  if ( tau_rise_AMPA >= tau_decay_AMPA )
  {
    throw BadParameter( "tau_rise_AMPA < tau_decay_AMPA required." );
  }
  if ( tau_rise_GABA_A >= tau_decay_GABA_A )
  {
    throw BadParameter( "tau_rise_GABA_A < tau_decay_GABA_A required." );
  }
  if ( tau_rise_GABA_B >= tau_decay_GABA_B )
  {
    throw BadParameter( "tau_rise_GABA_B < tau_decay_GABA_B required." );
  }
  if ( tau_rise_NMDA >= tau_decay_NMDA )
  {
    throw BadParameter( "tau_rise_NMDA < tau_decay_NMDA required." );
  }
  if ( tau_Mg_fast_NMDA >= tau_Mg_slow_NMDA )
  {
    throw BadParameter( "tau_Mg_fast_NMDA < tau_Mg_slow_NMDA required." );
  }

  if ( xmax_GABA_B1a > 1 )
  {
    throw BadProperty( "xmax_GABA_B1a <= 1 required.");
  }
  if ( xmax_GABA_B1a < 0 )
  {
    throw BadProperty( "xmax_GABA_B1a >= 0 required.");
  }
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::V_m, y_[ V_M ] );     // Membrane potential
  def< double >( d, names::theta, y_[ THETA ] ); // Threshold
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::State_::set( const DictionaryDatum& d, const ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike& node )
{
  updateValue< double >( d, names::V_m, y_[ V_M ] );
  updateValue< double >( d, names::theta, y_[ THETA ] );

  bool equilibrate = false;
  updateValue< bool >( d, names::equilibrate, equilibrate );
  if ( equilibrate )
  {
    y_[ m_fast_NMDA ] = node.m_eq_NMDA_( y_[ V_M ] );
    y_[ m_slow_NMDA ] = node.m_eq_NMDA_( y_[ V_M ] );
    y_[ m_Ih ] = node.m_eq_h_( y_[ V_M ] );
    y_[ State_::D_IKNa ] = node.D_eq_KNa_( y_[ V_M ] );
    y_[ m_IT ] = node.m_eq_T_( y_[ V_M ] );
    y_[ h_IT ] = node.h_eq_T_( y_[ V_M ] );
  }
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::Buffers_::Buffers_( ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike& n )
  : logger_( n )
  , spike_inputs_( std::vector< RingBuffer >( SUP_SPIKE_RECEPTOR - 1 ) )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
  , step_( Time::get_resolution().get_ms() )
  , integration_step_( step_ )
  , I_stim_( 0.0 )
{
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::Buffers_::Buffers_( const Buffers_&, ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike& n )
  : logger_( n )
  , spike_inputs_( std::vector< RingBuffer >( SUP_SPIKE_RECEPTOR - 1 ) )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
  , step_( Time::get_resolution().get_ms() )
  , integration_step_( step_ )
  , I_stim_( 0.0 )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike()
  : Archiving_Node()
  , P_()
  , S_( *this, P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike( const ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::~ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike()
{
  // GSL structs may not be initialized, so we need to protect destruction.
  if ( B_.e_ )
  {
    gsl_odeiv_evolve_free( B_.e_ );
  }
  if ( B_.c_ )
  {
    gsl_odeiv_control_free( B_.c_ );
  }
  if ( B_.s_ )
  {
    gsl_odeiv_step_free( B_.s_ );
  }
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::init_state_( const Node& proto )
{
  const ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike& pr = downcast< ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike >( proto );
  S_ = pr.S_;
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::init_buffers_()
{
  // Reset spike buffers.
  for ( std::vector< RingBuffer >::iterator it = B_.spike_inputs_.begin();
        it != B_.spike_inputs_.end();
        ++it )
  {
    it->clear(); // include resize
  }

  B_.currents_.clear(); // include resize

  // allocate strucure for gap events here
  // function is called from Scheduler::prepare_nodes() before the
  // first call to update
  // so we already know which interpolation scheme to use according
  // to the properties of this neurons
  // determine size of structure depending on interpolation scheme
  // and unsigned int Scheduler::min_delay() (number of simulation time steps
  // per min_delay step)

  // resize interpolation_coefficients depending on interpolation order
  const size_t buffer_size = kernel().connection_manager.get_min_delay()
    * ( kernel().simulation_manager.get_wfr_interpolation_order() + 1 );

  B_.interpolation_coefficients.resize( buffer_size, 0.0 );

  B_.last_y_values.resize( kernel().connection_manager.get_min_delay(), 0.0 );

  B_.sumj_g_ij_ = 0.0;
  //end gap junction stuff

  B_.logger_.reset();

  Archiving_Node::clear_history();

  B_.step_ = Time::get_resolution().get_ms();
  B_.integration_step_ = B_.step_;

  if ( B_.s_ == 0 )
  {
    B_.s_ =
      gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_step_reset( B_.s_ );
  }

  if ( B_.c_ == 0 )
  {
    B_.c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  }
  else
  {
    gsl_odeiv_control_init( B_.c_, 1e-3, 0.0, 1.0, 0.0 );
  }

  if ( B_.e_ == 0 )
  {
    B_.e_ = gsl_odeiv_evolve_alloc( State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_evolve_reset( B_.e_ );
  }

  B_.sys_.function = ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_dynamics;
  B_.sys_.jacobian = 0;
  B_.sys_.dimension = State_::STATE_VEC_SIZE;
  B_.sys_.params = reinterpret_cast< void* >( this );

  B_.I_stim_ = 0.0;
}

double
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_synapse_constant( double tau_1,
  double tau_2,
  double g_peak )
{
  /* The solution to the beta function ODE obtained by the solver is
   *
   *   g(t) = c / ( a - b ) * ( e^(-b t) - e^(-a t) )
   *
   * with a = 1/tau_1, b = 1/tau_2, a != b. The maximum of this function is at
   *
   *   t* = 1/(a-b) ln a/b
   *
   * We want to scale the function so that
   *
   *   max g == g(t*) == g_peak
   *
   * We thus need to set
   *
   *   c = g_peak * ( a - b ) / ( e^(-b t*) - e^(-a t*) )
   *
   * See Rotter & Diesmann, Biol Cybern 81:381 (1999) and Roth and van Rossum,
   * Ch 6, in De Schutter, Computational Modeling Methods for Neuroscientists,
   * MIT Press, 2010.
   */

  const double t_peak =
    ( tau_2 * tau_1 ) * std::log( tau_2 / tau_1 ) / ( tau_2 - tau_1 );

  const double prefactor = ( 1 / tau_1 ) - ( 1 / tau_2 );

  const double peak_value =
    ( std::exp( -t_peak / tau_2 ) - std::exp( -t_peak / tau_1 ) );

  return g_peak * prefactor / peak_value;
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::calibrate()
{
  // ensures initialization in case mm connected after Simulate
  B_.logger_.init();

  // The code below initializes conductance step size for incoming pulses.
  V_.cond_steps_.resize( SUP_SPIKE_RECEPTOR - 1 );

  V_.cond_steps_[ AMPA - 1 ] =
    get_synapse_constant( P_.tau_rise_AMPA, P_.tau_decay_AMPA, P_.g_peak_AMPA );

  V_.cond_steps_[ NMDA - 1 ] =
    get_synapse_constant( P_.tau_rise_NMDA, P_.tau_decay_NMDA, P_.g_peak_NMDA );

  V_.cond_steps_[ GABA_A - 1 ] = get_synapse_constant(
    P_.tau_rise_GABA_A, P_.tau_decay_GABA_A, P_.g_peak_GABA_A );

  V_.cond_steps_[ GABA_B - 1 ] = get_synapse_constant(
    P_.tau_rise_GABA_B, P_.tau_decay_GABA_B, P_.g_peak_GABA_B );

  V_.cond_steps_[ GABA_B1a - 1 ] = get_synapse_constant(
    P_.tau_rise_GABA_B1a, P_.tau_decay_GABA_B1a, P_.g_peak_GABA_B1a );

  V_.PotassiumRefractoryCounts_ = Time( Time::ms( P_.t_ref ) ).get_steps();

  V_.V_clamp_ = S_.y_[ State_::V_M ];
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d );
  Archiving_Node::get_status( d );

  DictionaryDatum receptor_type = new Dictionary();

  ( *receptor_type )[ names::AMPA ] = AMPA;
  ( *receptor_type )[ names::NMDA ] = NMDA;
  ( *receptor_type )[ names::GABA_A ] = GABA_A;
  ( *receptor_type )[ names::GABA_B ] = GABA_B;
  ( *receptor_type )[ "GABA_B1a" ] = GABA_B1a;
  ( *receptor_type )[ "MINI" ] = MINI; //added this

  ( *d )[ names::receptor_types ] = receptor_type;
  ( *d )[ names::recordables ] = recordablesMap_.get_list();
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, *this );  // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

bool
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::update_( nest::Time const& origin,
  const long from,
  const long to,
  const bool called_from_wfr_update )
{
  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  const size_t interpolation_order =
    kernel().simulation_manager.get_wfr_interpolation_order();
  const double wfr_tol = kernel().simulation_manager.get_wfr_tol();
  bool wfr_tol_exceeded = false;

  // allocate memory to store the new interpolation coefficients
  // to be sent by gap event
  const size_t buffer_size =
    kernel().connection_manager.get_min_delay() * ( interpolation_order + 1 );
  std::vector< double > new_coefficients( buffer_size, 0.0 );

  // parameters needed for piecewise interpolation
  double y_i = 0.0, y_ip1 = 0.0, hf_i = 0.0, hf_ip1 = 0.0;
  double f_temp[ State_::STATE_VEC_SIZE ];

  for ( long lag = from; lag < to; ++lag )
  {
    double tt = 0.0; // it's all relative!

    // B_.lag is needed by ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_dynamics to
    // determine the current section
    B_.lag_ = lag;

    // TODO WARNING I don't understand what this section is doing here...! (Tom)
    if ( called_from_wfr_update )
    {
      y_i = S_.y_[ State_::V_M ];
      if ( interpolation_order == 3 )
      {
        ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_dynamics(
          0, S_.y_, f_temp, reinterpret_cast< void* >( this ) );
        hf_i = B_.step_ * f_temp[ State_::V_M ];
      }
    }

    while ( tt < B_.step_ )
    {
      const int status = gsl_odeiv_evolve_apply( B_.e_,
        B_.c_,
        B_.s_,
        &B_.sys_,              // system of ODE
        &tt,                   // from t...
        B_.step_,              // ...to t=t+h
        &B_.integration_step_, // integration window (written on!)
        S_.y_ );               // neuron state

      if ( status != GSL_SUCCESS )
      {
        throw GSLSolverFailure( get_name(), status );
      }
    }

    // Enforce voltage clamp
    if ( P_.voltage_clamp )
    {
      assert( not called_from_wfr_update );
      S_.y_[ State_::V_M ] = V_.V_clamp_;
    }

    // Enforce instantaneous blocking of NMDA channels
    const double m_eq_NMDA = m_eq_NMDA_( S_.y_[ State_::V_M ] );
    S_.y_[ State_::m_fast_NMDA ] =
      std::min( m_eq_NMDA, S_.y_[ State_::m_fast_NMDA ] );
    S_.y_[ State_::m_slow_NMDA ] =
      std::min( m_eq_NMDA, S_.y_[ State_::m_slow_NMDA ] );

    if ( not called_from_wfr_update )
    {

      // A spike is generated if the neuron is not refractory and the membrane
      // potential exceeds the threshold.
      if ( S_.ref_steps_ == 0
        and S_.y_[ State_::V_M ] >= S_.y_[ State_::THETA ] )
      {
        // Set V and theta to the sodium reversal potential.
        S_.y_[ State_::V_M ] = P_.E_Na;
        S_.y_[ State_::THETA ] = P_.E_Na;

        // Activate fast re-polarizing potassium current. Add 1 to compensate
        // to subtraction right after while loop.
        S_.ref_steps_ = V_.PotassiumRefractoryCounts_ + 1;

        set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

        nest::SpikeEvent se;
        kernel().event_delivery_manager.send( *this, se, lag );
      }

      if ( S_.ref_steps_ > 0 )
      {
        --S_.ref_steps_;
      }

      /* Add arriving spikes.
       *
       * The input variable for the synapse type with buffer index i is
       * at position 2 + 2*i in the state variable vector.
       * At each spike times we add h0 * spike_weight * spike_multiplicity *
       * x_GABA_B1a to the d_DG/dt equation (dh/dt in Roth and Van Rossum 2009).
       * h0 is scaled such that the peak conductance is
       * (spike_weight * spike_multiplicity * x_GABA_B1a * g_peak).
       */
      for ( size_t i = 0; i < GABA_B1a - 1; ++i )
      {
        S_.y_[ 2 + 2 * i ] +=
          V_.cond_steps_[ i ] * B_.spike_inputs_[ i ].get_value( lag )
          * get_x_GABA_B1a( i );
      }

      /* Add arriving GABA_B1a spikes.
       *
       * We add a soft maximum to the value that the GABA_B1a conductance can
       * reach (or equivalently, a soft minimum to the value x_GABA_B1a can
       * reach for each receptor): for each incoming spike, the peak conductance
       * value of the GABA_B1a receptor is scaled by (g_max_GABA_B1a -
       * g_GABA_B1a)/g_max_GABA_B1a. This ensures that the GABA_B1a conductance is (softly)
       * bound g_max_GABA_B1a for small values of g_peak_GABA_B1a.
       * NB: In order to ensure the boundedness of g_peak_GABA_B1a even for high
       * values of g_peak_GABA_B1a, we also apply a hard bound in get_x_GABA_B1a.
       * At each spike times we add h0 * spike_weight * spike_multiplicity *
       * x_GABA_B1a * (g_max_GABA_B1a /- g_GABA_B1a) to the d_DG/dt equation (dh/dt
       * in Roth and Van Rossum 2009). h0 is scaled such that the peak
       * conductance is (spike_weight * spike_multiplicity * x_GABA_B1a * g_peak).
       */
      S_.y_[ 2 + 2 * ( GABA_B1a - 1 ) ] +=
        V_.cond_steps_[ GABA_B1a - 1 ] * B_.spike_inputs_[ GABA_B1a - 1 ].get_value( lag )
        * get_x_GABA_B1a( GABA_B1a - 1 ) *
        ( 1 - ( get_g_GABA_B1a() / P_.g_max_GABA_B1a ) );

      // Add MINI
      for ( size_t i = 0 ; i < B_.spike_inputs_[ MINI - 1 ].get_value( lag ) ; ++i )
      {
        S_.y_[ State_::V_M ] += P_.mini_mean + P_.mini_sigma *
           V_.normal_dev_( kernel().rng_manager.get_rng( get_thread() ) );
      }

      // set new input current
      B_.I_stim_ = B_.currents_.get_value( lag );

      B_.logger_.record_data( origin.get_steps() + lag );

    }
    else // if(called_from_wfr_update)
    {
      // SEND A SPIKE ONLY DURING FINAL UPDATE (-> weaker gap junction with no
      // effect of sharp variation in V_m during spike)
      // NOT SENDING SPIKE: Start
      // A spike is generated if the neuron is not refractory and the membrane
      // potential exceeds the threshold.
      // if ( S_.ref_steps_ == 0
      //   and S_.y_[ State_::V_M ] >= S_.y_[ State_::THETA ] )
      // {
        // Set V and theta to the sodium reversal potential.
        // S_.y_[ State_::V_M ] = P_.E_Na;
        // S_.y_[ State_::THETA ] = P_.E_Na;

        // Activate fast re-polarizing potassium current. Add 1 to compensate
        // to subtraction right after while loop.
        // S_.ref_steps_ = V_.PotassiumRefractoryCounts_ + 1;
        //
        // set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
        //
        // nest::SpikeEvent se;
        // kernel().event_delivery_manager.send( *this, se, lag );
      // }

      // We don't reduce the number of refractory steps during preliminary
      // updates
      // if ( S_.ref_steps_ > 0 )
      // {
      //   --S_.ref_steps_;
      // }
      // NOT SENDING SPIKE: end

      /* Add arriving spikes without deleting them from the buffer
       *
       * The input variable for the synapse type with buffer index i is
       * at position 2 + 2*i in the state variable vector.
       * At each spike times we add h0 * spike_weight * spike_multiplicity *
       * x_GABA_B1a to the d_DG/dt equation (dh/dt in Roth and Van Rossum 2009).
       * h0 is scaled such that the peak conductance is
       * (spike_weight * spike_multiplicity * x_GABA_B1a * g_peak).
       */
      for ( size_t i = 0; i < GABA_B1a - 1; ++i )
      {
        S_.y_[ 2 + 2 * i ] +=
          V_.cond_steps_[ i ] * B_.spike_inputs_[ i ].get_value_wfr_update( lag )
          * get_x_GABA_B1a( i );
      }

      /* Add arriving GABA_B1a spikes.
       *
       * We add a soft maximum to the value that the GABA_B1a conductance can
       * reach (or equivalently, a soft minimum to the value x_GABA_B1a can
       * reach for each receptor): for each incoming spike, the peak conductance
       * value of the GABA_B1a receptor is scaled by (g_max_GABA_B1a -
       * g_GABA_B1a)/g_max_GABA_B1a. This ensures that the GABA_B1a conductance is (softly)
       * bound g_max_GABA_B1a for small values of g_peak_GABA_B1a.
       * NB: In order to ensure the boundedness of g_peak_GABA_B1a even for high
       * values of g_peak_GABA_B1a, we also apply a hard bound in get_x_GABA_B1a.
       * At each spike times we add h0 * spike_weight * spike_multiplicity *
       * x_GABA_B1a * (g_max_GABA_B1a - g_GABA_B1a) to the d_DG/dt equation (dh/dt
       * in Roth and Van Rossum 2009). h0 is scaled such that the peak
       * conductance is (spike_weight * spike_multiplicity * x_GABA_B1a * g_peak).
       */
      S_.y_[ 2 + 2 * ( GABA_B1a - 1 ) ] +=
        V_.cond_steps_[ GABA_B1a - 1 ] * B_.spike_inputs_[ GABA_B1a - 1 ].get_value_wfr_update( lag )
        * get_x_GABA_B1a( GABA_B1a - 1 ) *
        ( 1 - ( get_g_GABA_B1a() / P_.g_max_GABA_B1a ) );

      // Add MINI
      for ( size_t i = 0 ; i < B_.spike_inputs_[ MINI - 1 ].get_value_wfr_update( lag ) ; ++i )
      {
        S_.y_[ State_::V_M ] += P_.mini_mean + P_.mini_sigma *
           V_.normal_dev_( kernel().rng_manager.get_rng( get_thread() ) );
      }


      // check if deviation from last iteration exceeds wfr_tol
      wfr_tol_exceeded = wfr_tol_exceeded
        or fabs( S_.y_[ State_::V_M ] - B_.last_y_values[ lag ] ) > wfr_tol;
      B_.last_y_values[ lag ] = S_.y_[ State_::V_M ];

      // update different interpolations

      // constant term is the same for each interpolation order
      new_coefficients[ lag * ( interpolation_order + 1 ) + 0 ] = y_i;

      switch ( interpolation_order )
      {
      case 0:
        break;

      case 1:
        y_ip1 = S_.y_[ State_::V_M ];

        new_coefficients[ lag * ( interpolation_order + 1 ) + 1 ] = y_ip1 - y_i;
        break;

      case 3:
        y_ip1 = S_.y_[ State_::V_M ];
        ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_dynamics(
          B_.step_, S_.y_, f_temp, reinterpret_cast< void* >( this ) );
        hf_ip1 = B_.step_ * f_temp[ State_::V_M ];

        new_coefficients[ lag * ( interpolation_order + 1 ) + 1 ] = hf_i;
        new_coefficients[ lag * ( interpolation_order + 1 ) + 2 ] =
          -3 * y_i + 3 * y_ip1 - 2 * hf_i - hf_ip1;
        new_coefficients[ lag * ( interpolation_order + 1 ) + 3 ] =
          2 * y_i - 2 * y_ip1 + hf_i + hf_ip1;
        break;

      default:
        throw BadProperty( "Interpolation order must be 0, 1, or 3." );
      }
    } // if (called_from_wfr_update)
  } // for lag = from; lag < to

  // if not called_from_wfr_update perform constant extrapolation
  // and reset last_y_values
  if ( not called_from_wfr_update )
  {
    for ( long temp = from; temp < to; ++temp )
    {
      new_coefficients[ temp * ( interpolation_order + 1 ) + 0 ] =
        S_.y_[ State_::V_M ];
    }

    std::vector< double >( kernel().connection_manager.get_min_delay(), 0.0 )
      .swap( B_.last_y_values );
  }

  // Send gap-event
  nest::GapJunctionEvent ge;
  ge.set_coeffarray( new_coefficients );
  kernel().event_delivery_manager.send_secondary( *this, ge );

  // Reset variables
  B_.sumj_g_ij_ = 0.0;
  std::vector< double >( buffer_size, 0.0 )
    .swap( B_.interpolation_coefficients );

  return wfr_tol_exceeded;

} // update_

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::handle( nest::SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );
  assert( e.get_rport() < static_cast< int >( B_.spike_inputs_.size() ) );

  B_.spike_inputs_[ e.get_rport() ].add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_multiplicity() );
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  const double I = e.get_current();
  const double w = e.get_weight();

  // add weighted current; HEP 2002-10-04
  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    w * I );
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::handle( nest::DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

void
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike::handle( nest::GapJunctionEvent& e )
{

  B_.sumj_g_ij_ += e.get_weight();

  size_t i = 0;
  std::vector< unsigned int >::iterator it = e.begin();
  // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  while ( it != e.end() )
  {
    B_.interpolation_coefficients[ i ] +=
      e.get_weight() * e.get_coeffvalue( it );
    ++i;
  }
}

#endif // HAVE_GSL
