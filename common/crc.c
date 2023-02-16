/*++
 *
 * Copyright (c) 2004-2006 Intel Corporation - All Rights Reserved
 *
 * This software program is licensed subject to the BSD License, 
 * available at http://www.opensource.org/licenses/bsd-license.html
 *
 * Abstract: The main routine
 * 
 --*/

#include <stdio.h>
#include "crc.h"
#include "stdlib.h"
#include "string.h"


/* 
 * the following variables are used for counting cycles and bytes 
 */

extern uint32_t crc_tableil8_o32[256];
extern uint32_t crc_tableil8_o40[256];
extern uint32_t crc_tableil8_o48[256];
extern uint32_t crc_tableil8_o56[256];
extern uint32_t crc_tableil8_o64[256];
extern uint32_t crc_tableil8_o72[256];
extern uint32_t crc_tableil8_o80[256];
extern uint32_t crc_tableil8_o88[256];

/**
 *
 * Routine Description:
 *
 * Computes the CRC32c checksum for the specified buffer.                      
 *
 * Arguments:
 *
 *		p_running_crc - pointer to the initial or final remainder value 
 *						used in CRC computations. It should be set to 
 *						non-NULL if the mode argument is equal to CONT or END
 *		p_buf - the packet buffer where crc computations are being performed
 *		length - the length of p_buf in bytes
 *		init_bytes - the number of initial bytes that need to be procesed before
 *					 aligning p_buf to multiples of 4 bytes
 *		mode - can be any of the following: BEGIN, CONT, END, BODY, ALIGN 
 *
 * Return value:
 *		
 *		The computed CRC32c value
 */

uint32_t
crc32c(
	uint32_t*		p_running_crc,
    const uint8_t*	p_buf,
    const uint32_t	length,
	uint8_t			mode) 
{
	uint32_t crc;
    const uint8_t* p_end = p_buf + length;
	if( (mode == MODE_CONT) || (mode == MODE_END) )
    crc = *p_running_crc;
  else
		crc = CRC32C_INIT_REFLECTED;
	while(p_buf < p_end )
		crc = crc_tableil8_o32[(crc ^ *p_buf++) & 0x000000FF] ^ (crc >> 8);
	if((mode == MODE_BEGIN) || (mode == MODE_CONT))
		return crc;		
	return crc ^ XOROT;

}


/**
 *
 * Routine Description:
 *
 * Computes the CRC32c checksum for the specified buffer using the slicing by 8 
 * algorithm over 64 bit quantities.                      
 *
 * Arguments:
 *
 *		p_running_crc - pointer to the initial or final remainder value 
 *						used in CRC computations. It should be set to 
 *						non-NULL if the mode argument is equal to CONT or END
 *		p_buf - the packet buffer where crc computations are being performed
 *		length - the length of p_buf in bytes
 *		init_bytes - the number of initial bytes that need to be procesed before
 *					 aligning p_buf to multiples of 4 bytes
 *		mode - can be any of the following: BEGIN, CONT, END, BODY, ALIGN 
 *
 * Return value:
 *		
 *		The computed CRC32c value
 */

uint32_t
crc32c_sb8_64_bit(
	uint32_t* p_running_crc,
    const uint8_t*	p_buf,
    const uint32_t length,
	const uint32_t init_bytes,
	uint8_t			mode)
{
	uint32_t li;
	uint32_t crc, term1, term2;
	uint32_t running_length;
	uint32_t end_bytes;
	if( (mode ==  MODE_CONT) || (mode == MODE_END))
    crc = *p_running_crc;
  else
		crc = CRC32C_INIT_REFLECTED;
	running_length = ((length - init_bytes)/8)*8;
	end_bytes = length - init_bytes - running_length; 

	for(li=0; li < init_bytes; li++) 
		crc = crc_tableil8_o32[(crc ^ *p_buf++) & 0x000000FF] ^ (crc >> 8);	
	for(li=0; li < running_length/8; li++) 
	{
		crc ^= *(uint32_t *)p_buf;
		p_buf += 4;
		term1 = crc_tableil8_o88[crc & 0x000000FF] ^
				crc_tableil8_o80[(crc >> 8) & 0x000000FF];
		term2 = crc >> 16;
		crc = term1 ^
			  crc_tableil8_o72[term2 & 0x000000FF] ^ 
			  crc_tableil8_o64[(term2 >> 8) & 0x000000FF];
		term1 = crc_tableil8_o56[(*(uint32_t *)p_buf) & 0x000000FF] ^
				crc_tableil8_o48[((*(uint32_t *)p_buf) >> 8) & 0x000000FF];
		
		term2 = (*(uint32_t *)p_buf) >> 16;
		crc =	crc ^ 
				term1 ^		
				crc_tableil8_o40[term2  & 0x000000FF] ^	
				crc_tableil8_o32[(term2 >> 8) & 0x000000FF];	
		p_buf += 4;
	}
	for(li=0; li < end_bytes; li++) 
		crc = crc_tableil8_o32[(crc ^ *p_buf++) & 0x000000FF] ^ (crc >> 8);
	if((mode == MODE_BEGIN) || (mode ==  MODE_CONT))
		return crc;		
    return crc ^ XOROT;	
}
