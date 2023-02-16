/*++
 *
 * Copyright (c) 2004-2006 Intel Corporation - All Rights Reserved
 * 
 * This software program is licensed subject to the BSD License, 
 * available at http://www.opensource.org/licenses/bsd-license.html
 *
 --*/

#include <stdint.h>

#ifndef CRC_H_INCLUDED
#define CRC_H_INCLUDED

#define TEST_CRC_VERSION			"BUILD 2"
#define CPU_DATA_CACHE_SIZE			0x100000
#define WARM		1
#define	COLD		2
#define RANDOM		3
#define	CONSTANT	4
#define	INCREMENTAL	5
#define	INIT_TABLE_STATUS		WARM	
#define INIT_DATA_STATUS		WARM
#define	INIT_NUM_OF_ITERATIONS	10
#define	INIT_PACKET_SIZE		1024
#define	INIT_ITERATION_STYLE	CONSTANT
#define	INIT_ALIGNMENT			0
#define	INIT_ALIGNMENT_STYLE	CONSTANT
#define MAX_BUF_SIZE			65536
#define MIN_BUF_SIZE			64
#define PACKET_SIZE_INCREMENT	64
#define MAX_ALIGNMENT			8
#define MIN_ALIGNMENT			1
#define ALIGNMENT_INCREMENT		1
#define MPA_FRAME_LENGTH		48
#define MPA_FRAME_INDEX1		5
#define MPA_FRAME_VALUE1		0x2a
#define MPA_FRAME_INDEX2		6
#define MPA_FRAME_VALUE2		0x40
#define MPA_FRAME_INDEX3		7
#define MPA_FRAME_VALUE3		0x03
#define MPA_FRAME_INDEX4		19
#define MPA_FRAME_VALUE4		0x01
#define MPA_FRAME_CRC			0x84B3864C
#define UINT8_MAX				255
#define LONG_WORD_SIZE			4
#define	TRUE					1
#define	FALSE					0
#define CRC_FAILED				1
#define CRC_PASSED				0
#define CRC32C_INIT_REFLECTED 0xFFFFFFFF
#define XOROT 0xFFFFFFFF
#define MODE_BEGIN	0
#define	MODE_CONT	1
#define	MODE_END	2
#define	MODE_BODY	3
#define	MODE_ALIGN	4
#define TWO_CORE_TAIL_LENGTH	16
#define FOUR_CORE_TAIL_LENGTH	32
#define SLICE_LENGTH			8
#define SB8_CHUNK				8
#define POWER_OF_2(X) (1 << (X))
#define MAX_SLICES	8
#define MAX_CHARS	100
#define INIT_WIDTH			32				
#define INIT_POLY			0x1EDC6F41L		
#define INIT_REFLECTED		TRUE	
#define INIT_SLICE_LENGTH   8
#define INIT_NUM_OF_SLICES	8
#define INIT_OFFSET			32
#define INIT_DIR			".\\"	
#define	INIT_FILE			"8x256_tables.c"
#define SB3_1_SLICE_1		10
#define SB3_1_SLICE_2		10
#define SB3_1_SLICE_3		12
#define SB3_NUM_OF_SLICES	3
#define	SB3_1_FILE			"4K_plus_2x1K_tables.c"
#define SB3_2_SLICE_1		10
#define SB3_2_SLICE_2		11
#define SB3_2_SLICE_3		11
#define	SB3_2_FILE			"1K_plus_2x2K_tables.c"
#define SB2_NUM_OF_SLICES	2
#define SB2_SLICE_1			16
#define SB2_SLICE_2			16
#define	SB2_FILE			"2x64K_tables.c"
#define SB1_NUM_OF_SLICES	1
#define	SB1_FILE			"256_table.c"

#ifdef __cplusplus
extern "C"
{
#endif	/* __cplusplus */

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
	uint8_t			mode);


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
	uint8_t			mode);

#ifdef __cplusplus
}
#endif	/* __cplusplus */

#endif	//CRC_H_INCLUDED

