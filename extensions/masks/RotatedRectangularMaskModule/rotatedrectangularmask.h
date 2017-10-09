#ifndef ROTATED_RECTANGULAR_MASK_H
#define ROTATED_RECTANGULAR_MASK_H

// Includes from nestkernel:
#include "mask.h"

// Includes from topology:
#include "topology_names.h"

namespace spiking_visnet
{

class RotatedRectangularMask : public nest::Mask< 2 >
{
public:
  /**
   * Parameters that should be in the dictionary:
   * lower_left  - Position of lower left corner (array of doubles)
   * upper_right - Position of upper right corner (array of doubles)
   * angle - Counter-clockwise rotation angle of the rectangle (double)
   */
  RotatedRectangularMask( const DictionaryDatum& d ):
    angle_( 0.0 )
    {

      lower_left_ = getValue< std::vector< double > >(
          d, nest::names::lower_left );
      upper_right_ = getValue< std::vector< double > >(
          d, nest::names::upper_right );
      if ( not( lower_left_ < upper_right_ ) )
        throw nest::BadProperty(
            "topology::RotatedRectangularMask: "
            "Upper right must be strictly to the right and above lower left." );

      // NOTE: must be in radians
      updateValue< double >( d, "angle", angle_ );

      // precompute bounding box coordinates:
      // build upper-left and lower-right
      nest::Position< 2 > upper_left_( lower_left_[ 0 ], upper_right_[ 1 ] );
      nest::Position< 2 > lower_right_( upper_right_[ 0 ], lower_left_[ 1 ] );
      // rotate all corners
      nest::Position< 2 > rotated_ll = rotate( lower_left_,  angle_ );
      nest::Position< 2 > rotated_ur = rotate( upper_right_, angle_ );
      nest::Position< 2 > rotated_ul = rotate( upper_left_,  angle_ );
      nest::Position< 2 > rotated_lr = rotate( lower_right_, angle_ );
      // get minimum of all x and y coordinates
      double x_coords[] = {
        rotated_ll[ 0 ], rotated_ur[ 0 ], rotated_ul[ 0 ], rotated_lr[ 0 ]};
      double y_coords[] = {
        rotated_ll[ 1 ], rotated_ur[ 1 ], rotated_ul[ 1 ], rotated_lr[ 1 ]};
      double *min_rotated_x = std::min_element( x_coords, x_coords + 4 );
      double *min_rotated_y = std::min_element( y_coords, y_coords + 4 );
      double *max_rotated_x = std::max_element( x_coords, x_coords + 4 );
      double *max_rotated_y = std::max_element( y_coords, y_coords + 4 );
      // lower-left: (min of all x coordinates, min of all y coordinates)
      // upper-right: (max of all x coordinates, max of all y coordinates)
      bb_lower_left = nest::Position< 2 >( *min_rotated_x, *min_rotated_y );
      bb_upper_right = nest::Position< 2 >( *max_rotated_x, *max_rotated_y );

    }

  ~RotatedRectangularMask()
  {
  }

  // returns a new point rotated COUNTERCLOCKWISE by the given angle
  nest::Position< 2 > rotate( const nest::Position< 2 > &p, double theta ) const
    {
      double x = p[ 0 ] * std::cos( theta ) - p[ 1 ] * std::sin( theta );
      double y = p[ 0 ] * std::sin( theta ) + p[ 1 ] * std::cos( theta );
      return nest::Position< 2 >( x, y );
    }

  using nest::Mask< 2 >::inside;

  /**
   * @returns true if point is inside the box
   */
  bool inside( const nest::Position< 2 >& p ) const
  {
    // apply the inverse rotation to the point (note negative angle)
    nest::Position< 2 > inverse_p = rotate( p, ( - angle_ ));
    // check if the rotated point is within the unrotated rectangle
    return ( inverse_p >= lower_left_ ) && ( inverse_p <= upper_right_ );
  }

  /**
   * @returns true if the whole given box is inside this box
   */
  bool inside( const nest::Box< 2 >& b ) const
  {
    nest::Position< 2 > p = b.lower_left;

    // Test if all corners are inside mask

    if ( not inside( p ) )
      return false; // (0, 0)
    p[ 0 ] = b.upper_right[ 0 ];
    if ( not inside( p ) )
      return false; // (0, 1)
    p[ 1 ] = b.upper_right[ 1 ];
    if ( not inside( p ) )
      return false; // (1, 1)
    p[ 0 ] = b.lower_left[ 0 ];
    if ( not inside( p ) )
      return false; // (1, 0)

    return true;
  }

  /**
   * @returns true if the whole given box is outside this box
   */
  using nest::Mask< 2 >::outside;

  /**
   * @returns a bounding box for the mask
   */
  nest::Box< 2 > get_bbox() const
  {
    return nest::Box< 2 >( bb_lower_left, bb_upper_right );
  }

  DictionaryDatum get_dict() const
  {
    DictionaryDatum d( new Dictionary );
    DictionaryDatum maskd( new Dictionary );
    def< DictionaryDatum >( d, get_name(), maskd );
    def< std::vector< double > >(
            maskd, nest::names::lower_left, lower_left_ );
    def< std::vector< double > >(
            maskd, nest::names::upper_right, upper_right_ );
    def< double >(
            maskd, "angle", angle_ );
    return d;
  }

  nest::Mask< 2 >* clone() const
  {
    return new RotatedRectangularMask( *this );
  }

  /**
   * @returns the name of this mask type.
   */
  static Name get_name()
  {
      return "rotated_rectangular";
  }

protected:
  nest::Position< 2 > lower_left_;
  nest::Position< 2 > upper_right_;
  double angle_;
  nest::Position< 2 > bb_lower_left;
  nest::Position< 2 > bb_upper_right;
}; // class RotatedRectangularMask

} // namespace spiking_visnet

#endif
