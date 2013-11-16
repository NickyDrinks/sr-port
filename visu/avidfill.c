;/****************************************************************************
;** MODULE:	avidfill.asm
;** AUTHOR:	Sami Tammilehto / Fennosoftec OY
;** DOCUMENT:	?
;** VERSION:	1.0
;** REFERENCE:	-
;** REVISED BY:	-
;*****************************************************************************
;**
;** Assembler / Video (polygon filling)
;**
;****************************************************************************/

/*
;�������� _vid_drawpolylist(...) ��������
;This call is obsolote. The correct call is to _draw_polylist. 
_vid_drawpolylist PROC FAR
	pop	cx ;retoffset
	pop	dx ;retsegment
	mov	ax,0f001h
	push	ax ;oflags
	push	dx ;retsegment
	push	cx ;retoffset
	jmp	_draw_polylist ;draw.asm
_vid_drawpolylist ENDP
*/

/*
;�������� _vid_drawfill(char *drawdata) ��������
;entry:	[drawdata]=pointer to fill data stream
; exit: -
;descr: This routine fills a polygon on screen according to instructions in
;       the data stream. The 'nrm' stands for normal = flat shaded polygon.
;	Stream format: {FILL-DATA}
drawfill_progs LABEL WORD
dw OFFSET drawfill_nrm
dw OFFSET drawfill_grd
_vid_drawfill PROC FAR
	CBEG
	call	vidstart
	lfspar	si,0
	mov	bx,ds:[si]
	cmp	bx,1
	ja	@@1
	add	bx,bx
	add	si,2
	call	drawfill_progs[bx]
@@1:	CEND
_vid_drawfill ENDP
*/

void vid_drawfill(char *drawdata)
{
}

/*
ALIGN	4
leftside   db	1111b,1110b,1100b,1000b
rightside  db	0001b,0011b,0111b,1111b
middledot  db	0001b,0010b,0100b,1000b
ALIGN 16
vdf_leftx	dd	0
vdf_lefta	dd	0
vdf_rightx	dd	0
vdf_righta	dd	0
vdf_count	dw	0
vdf_di		dw	0
vdf_color	db	0
ALIGN 16
vdfg_color1	dw	0 
vdfg_color2	dw	0
vdfg_left	dw	0
vdfg_right	dw	0
vdfg_leftc	dw	0 ;c & ca must be in this order for 32 bit 
vdfg_leftca	dw	0 ;data transfer (seek for [32])
vdfg_rightc	dw	0
vdfg_rightca	dw	0
vdfg_lefts	dw	0
vdfg_left4	dw	0
vdfg_buf	db	MAXCOLS dup(0)

;�������� drawfill_nrm ��������
;entry:	DS:SI=pointer to {NORMAL-FILL-DATA} stream
; exit: -
drawfill_nrm PROC NEAR
	mov	ax,fs:[si]
	mov	cs:vdf_color,al
	mov	bx,fs:[si+2] ;StartY
	add	si,4
	shl	bx,1
	mov	ax,ds:_rows[bx]
	mov	cs:vdf_di,ax
	;es:vdf_di=dest
	jmp	@@1	
@@0:	ret

@@1:	mov	ax,fs:[si]
	add	si,2
	or	ax,ax
	js	@@0
	jz	@@11
	mov	eax,fs:[si]
	mov	cs:vdf_leftx,eax
	mov	eax,fs:[si+4]
	mov	cs:vdf_lefta,eax
	add	si,8
@@11:	mov	ax,fs:[si]
	add	si,2
	or	ax,ax
	js	@@0
	jz	@@12
	mov	eax,fs:[si]
	mov	cs:vdf_rightx,eax
	mov	eax,fs:[si+4]
	mov	cs:vdf_righta,eax
	add	si,8
@@12:	mov	ax,fs:[si]
	or	ax,ax
	jz	@@0 ;ERROR
	add	si,2
	mov	cs:vdf_count,ax
	mov	di,cs:vdf_di
@@21:	;Fill loop
	;;
	mov	eax,cs:vdf_lefta
	add	cs:vdf_leftx,eax
	mov	eax,cs:vdf_righta
	add	cs:vdf_rightx,eax
	;
	mov	dx,word ptr cs:vdf_leftx[2]
	mov	ax,word ptr cs:vdf_rightx[2]
	;dx=leftx, ax=rightx
	;
	cmp	ax,dx
	je	@@20
	jl	@@22
	xchg	ax,dx
@@22:	dec	dx ;don't fill the rightmost pixel

	mov	bx,ax
	sar	bx,2
	mov	cx,dx
	sar	cx,2
	sub	cx,bx
	add	di,bx

	mov	bp,3
	and	bp,ax
	mov	bh,cs:leftside[bp]
	mov	bp,3
	and	bp,dx
	mov	bl,cs:rightside[bp]
	mov	dx,3c5h
	
	;(di..si,bx)
	cmp	cx,0
	je	@@29 ;end and beg in same byte
	;left side
	mov	al,bh
	out	dx,al
	mov	al,cs:vdf_color
	mov	es:[di],al
	inc	di
	dec	cx
	mov	ah,al
	;middle
	jcxz	@@24
	mov	al,0fh
	out	dx,al
	mov	al,ah
	mov	bp,bx
	call	faststosb
	mov	bx,bp
@@24:	;right side
	mov	al,bl
	out	dx,al
	mov	al,ah
	mov	es:[di],al
	jmp	@@20
@@29:	;end and beg in same byte
	mov	al,bl
	and	al,bh
	out	dx,al
	mov	al,cs:vdf_color
	mov	es:[di],al
	;;
@@20:	mov	di,cs:vdf_di
	add	di,ds:_rowlen
	mov	cs:vdf_di,di
	dec	cs:vdf_count
	jnz	@@21
	jmp	@@1
drawfill_nrm ENDP

;�������� drawfill_grd ��������
;entry:	DS:SI=pointer to {GOURAUD-FILL-DATA} stream
; exit: -
drawfill_grd PROC NEAR
	mov	bx,fs:[si+2] ;StartY
	add	si,4
	shl	bx,1
	mov	ax,ds:_rows[bx]
	mov	cs:vdf_di,ax
	;es:vdf_di=dest
	jmp	@@1	
@@0:	ret

@@1:	mov	ax,fs:[si]
	add	si,2
	or	ax,ax
	js	@@0
	jz	@@11
	mov	eax,fs:[si] ;c&ca
	mov	dword ptr cs:vdfg_leftc,eax ;[32]
	mov	eax,fs:[si+4]
	mov	cs:vdf_leftx,eax
	mov	eax,fs:[si+8]
	mov	cs:vdf_lefta,eax
	add	si,12
@@11:	mov	ax,fs:[si]
	add	si,2
	or	ax,ax
	js	@@0
	jz	@@12
	mov	eax,fs:[si] ;c&ca
	mov	dword ptr cs:vdfg_rightc,eax ;[32]
	mov	eax,fs:[si+4]
	mov	cs:vdf_rightx,eax
	mov	eax,fs:[si+8]
	mov	cs:vdf_righta,eax
	add	si,12
@@12:	mov	ax,fs:[si]
	or	ax,ax
	jz	@@0 ;ERROR
	add	si,2
	mov	cs:vdf_count,ax
	mov	di,cs:vdf_di
@@21:	;Fill loop
	;;
	mov	ax,cs:vdfg_leftca
	add	cs:vdfg_leftc,ax
	mov	ax,cs:vdfg_rightca
	add	cs:vdfg_rightc,ax
	mov	eax,cs:vdf_lefta
	add	cs:vdf_leftx,eax
	mov	eax,cs:vdf_righta
	add	cs:vdf_rightx,eax
	;
	mov	dx,word ptr cs:vdf_leftx[2]
	mov	ax,word ptr cs:vdf_rightx[2]
	mov	bx,word ptr cs:vdfg_leftc
	mov	cx,word ptr cs:vdfg_rightc
	;dx=leftx, ax=rightx
	;
	cmp	ax,dx
	je	@@20
	jl	@@22
	xchg	bx,cx
	xchg	ax,dx
@@22:	dec	dx ;don't fill the rightmost pixel
	mov	cs:vdfg_color1,cx
	mov	cs:vdfg_color2,bx
	mov	cs:vdfg_left,ax
	mov	cs:vdfg_right,dx
	;set endpoint colors
	mov	dx,3c5h
	;end
	mov	bl,byte ptr cs:vdfg_right
	test	bl,1
	jnz	@@ef2
	and	bx,3
	mov	al,cs:middledot[bx]
	out	dx,al
	mov	bx,cs:vdfg_right
	shr	bx,2
	add	bx,di
	mov	al,byte ptr cs:vdfg_color2[1]
	mov	es:[bx],al
@@ef2:	;start
	mov	bl,byte ptr cs:vdfg_left
	test	bl,1
	jz	@@ef1
	and	bx,3
	mov	al,cs:middledot[bx]
	out	dx,al
	mov	bx,cs:vdfg_left
	shr	bx,2
	add	bx,di
	mov	al,byte ptr cs:vdfg_color1[1]
	mov	es:[bx],al
@@ef1:	;calc length & beg
	mov	ax,cs:vdfg_left
	mov	dx,cs:vdfg_right
	inc	ax
	shr	ax,1
	mov	cs:vdfg_lefts,ax
	dec	dx
	shr	dx,1
	;
	mov	cx,ax
	shr	cx,1
	add	di,cx
	;
	mov	cx,dx
	sub	cx,ax
	inc	cx ;cx=number of 2 byte blocks to fill
	cmp	cx,0
	jg	@@2f2
	;ends only
	jmp	@@20
@@2f2:
	push	si
	mov	ax,cs:vdfg_color1
	sub	ax,cs:vdfg_color2
	jge	@@2f3
	neg	ax
	mov	bx,cx
	add	bx,bx
	mul	word ptr ds:_afilldiv[bx]
	neg	dx
	jmp	@@2f4
@@2f3:	mov	bx,cx
	add	bx,bx
	mul	word ptr ds:_afilldiv[bx]
@@2f4:	mov	si,cx

@@d0:	;dx=coloradder, cx=len (in 2 byte blocks)
	cmp	dx,128
	jge	gouraud12
	cmp	dx,-128
	jle	gouraud12
	jmp	gouraud4
gouraud12:
	cmp	dx,400
	jge	gouraud1
	cmp	dx,-400
	jle	gouraud1
	jmp	gouraud2

;----------------------------------------------------------------

gouraud1 PROC NEAR ;1 pixel accuray
	sar	dx,1
	add	cx,cx
	;
	mov	ax,cx
	shl	cx,3
	sub	ax,cx
	;ax=-cx*7
	add	ax,OFFSET @@vdfg_fcode
	mov	bx,cs:vdfg_color2
	jmp	ax
	
zzz=MAXCOLS
REPT	MAXCOLS+1
	add	bx,dx			;2 bytes
	mov	cs:vdfg_buf[ (zzz and 3)*(MAXCOLS/4) + (zzz shr 2) ],bh	;5 bytes
	zzz=zzz-1
ENDM
	mov ax,bx			;2 bytes, 1 clock (filler)
@@vdfg_fcode:
	push	ds
	mov	ax,cs
	mov	ds,ax
	mov	cx,si
	add	cx,cx
	;
@@f1:	mov	al,00010001b
	test	cs:vdfg_lefts,1
	jz	@@f2
	mov	al,01000100b
@@f2:	;
	zzz=0
	add	cx,3
	REPT 4
	local	l1
	push	cx
	shr	cx,2
	jz	l1
	push	ax
	push	di
	mov	dx,3c5h
	out	dx,al
	mov	si,OFFSET vdfg_buf+zzz*(MAXCOLS/4)
	call	fastmovsb
	pop	di
	pop	ax
l1:	pop	cx
	dec	cx
	rol	al,1
	adc	di,0
	zzz=zzz+1
	ENDM
	;
@@f0:	pop	ds
	pop	si
	jmp	zzz20
gouraud1 ENDP

;----------------------------------------------------------------

gouraud2 PROC NEAR ;2 pixel accuray
	mov	ax,cx
	shl	cx,3
	sub	ax,cx
	;ax=-cx*7
	add	ax,OFFSET @@vdfg_fcode
	mov	bx,cs:vdfg_color2
	jmp	ax
	
zzz=MAXCOLS
REPT	MAXCOLS+1
	add	bx,dx			;2 bytes
	mov	cs:vdfg_buf[ (zzz and 1)*(MAXCOLS/2) + (zzz shr 1) ],bh	;5 bytes
	zzz=zzz-1
ENDM
	mov ax,bx			;2 bytes, 1 clock (filler)
@@vdfg_fcode:
	push	ds
	mov	ax,cs
	mov	ds,ax
	mov	cx,si
	jcxz	@@f0
	;
@@f1:	mov	al,00110011b
	test	cs:vdfg_lefts,1
	jz	@@f2
	mov	al,11001100b
@@f2:	;
	mov	si,OFFSET vdfg_buf
	push	cx
	push	di
	push	ax
	inc	cx
	shr	cx,1
	mov	dx,3c5h
	out	dx,al
	call	fastmovsb
	pop	ax
	pop	di
	pop	cx
	rol	al,2
	jnc	@@f3
	inc	di
@@f3:	mov	si,OFFSET vdfg_buf+MAXCOLS/2
	shr	cx,1
	jz	@@f0
	mov	dx,3c5h
	out	dx,al
	call	fastmovsb
	;
@@f0:	pop	ds
	pop	si
	jmp	zzz20
gouraud2 ENDP

;----------------------------------------------------------------

gouraud4 PROC NEAR ;4 pixel accuray
	cmp	cx,4
	jl	gouraud2
	sal	dx,1
	;
	mov	cs:vdfg_left4,cx
	shr	cx,1 ;rescale
	inc	cx ;one extra
	mov	ax,cx
	shl	cx,3
	sub	ax,cx
	;ax=-cx*7
	add	ax,OFFSET @@vdfg_fcode
	mov	bx,cs:vdfg_color2
	sub	bx,dx ;one extra
	jmp	ax
	
zzz=MAXCOLS
REPT	MAXCOLS+1
	add	bx,dx			;2 bytes
	mov	cs:vdfg_buf[zzz],bh	;5 bytes
	zzz=zzz-1
ENDM
	mov ax,bx			;2 bytes, 1 clock (filler)
@@vdfg_fcode: 
	push	ds
	mov	dx,cs
	mov	ds,dx
	mov	cx,si
	;
@@f1:	mov	si,OFFSET vdfg_buf
	test	cs:vdfg_lefts,1
	jnz	@@f3
	inc	cx
@@f4:	shr	cx,1
	jnc	@@f5
	mov	dx,3c5h
	mov	al,0fh
	out	dx,al
	call	fastmovsb
@@f0:	pop	ds
	pop	si
	jmp	zzz20

@@f5:	dec	cx
	mov	dx,3c5h
	mov	al,0fh
	out	dx,al
	call	fastmovsb
	mov	dx,3c5h
	mov	al,03h
	out	dx,al
	movsb
	pop	ds
	pop	si
	jmp	zzz20

@@f3:	mov	dx,3c5h
	mov	al,0ch
	out	dx,al
	movsb
	jmp	@@f4
gouraud4 ENDP

;----------------------------------------------------------------
	
zzz20:	;;
@@20:	mov	di,cs:vdf_di
	add	di,ds:_rowlen
	mov	cs:vdf_di,di
	dec	cs:vdf_count
	jnz	@@21
	jmp	@@1
drawfill_grd ENDP

;�������� fastmovsb ��������
;entry:	DS:SI=pointer to source data
;	ES:DI=pointer to destination (screen)
;	CX=number of bytes to copy (MUST BE <512*4)
;exit:	copies data like REP MOVSB but uses 32 bit moves and aligns
;	destination to dword boundary.
ALIGN 16
fastmovsb_bt LABEL WORD
	dw	OFFSET fastmovsb_b0
	dw	OFFSET fastmovsb_b1
	dw	OFFSET fastmovsb_b2
	dw	OFFSET fastmovsb_b3
fastmovsb_b1:
	mov	al,ds:[si]
	sub	cx,3
	mov	es:[di],al
	mov	ax,ds:[si+1]
	add	si,3
	mov	es:[di+1],ax
	add	di,3
	jmp	fastmovsb_b0
fastmovsb_b2:
	mov	ax,ds:[si]
	sub	cx,2
	add	si,2
	mov	es:[di],ax
	add	di,2
	jmp	fastmovsb_b0
fastmovsb_b3:
	mov	al,ds:[si]
	dec	cx
	inc	si
	mov	es:[di],al
	inc	di
	jmp	fastmovsb_b0
ALIGN 16
fastmovsb_et LABEL WORD
	dw	OFFSET fastmovsb_e0
	dw	OFFSET fastmovsb_e1
	dw	OFFSET fastmovsb_e2
	dw	OFFSET fastmovsb_e3
fastmovsb_e1:
	mov	al,ds:[si]
	inc	si
	mov	es:[di],al
	inc	di
	ret
fastmovsb_e3:
	mov	ax,ds:[si]
	mov	es:[di],ax
	mov	al,ds:[si+2]
	add	si,3
	mov	es:[di+2],al
	add	di,3
	ret
fastmovsb_e2:
	mov	ax,ds:[si]
	add	si,2
	mov	es:[di],ax
	add	di,2
fastmovsb_e0:
	ret
ALIGN 16
fastmovsb PROC NEAR
	;copies CX bytes of data from DS:SI to ES:DI aligning destination
	;si/di will point to the correct position after this (cx won't)
	cmp	cx,4
	jb	@@4
	mov	bx,di
	and	bx,3
	add	bx,bx
	jmp	fastmovsb_bt[bx]
fastmovsb_b0:
	mov	bx,cx
	add	bx,bx
	add	bx,bx
	and	bx,not 15
	neg	bx
	;bx=-cx*8
	add	bx,OFFSET @@jmp
	jmp	bx
	REPT	512
	db	66h,8bh,84h,00h,00h	;mov	eax,ds:[si+0000h]
	db	83h,0c6h,04h		;add	si,4
	db	66h,26h,89h,05h		;mov	es:[di],eax
	db	81h,0c7h,04h,00h	;add	di,0004h
	ENDM	;total 16 bytes
@@jmp:	mov	bx,cx
	and	bx,3
	add	bx,bx
	jmp	fastmovsb_et[bx]
@@4:	mov	bx,cx
	add	bx,bx
	jmp	fastmovsb_et[bx]
fastmovsb ENDP

;�������� faststosb ��������
;entry:	ES:DI=pointer to destination (screen)
;	AX=word fill (lo/hi byte MUST be same)
;	CX=number of bytes to fill (MUST BE <512*4)
;exit:	fills data like REP STOSB but uses 32 bit moves and aligns
;	destination to dword boundary.
;	NOTE: BX modified
ALIGN 16
faststosb_bt LABEL WORD
	dw	OFFSET faststosb_b0
	dw	OFFSET faststosb_b1
	dw	OFFSET faststosb_b2
	dw	OFFSET faststosb_b3
faststosb_b1:
	mov	es:[di],al
	sub	cx,3
	mov	es:[di+1],ax
	add	di,3
	jmp	faststosb_b0
faststosb_b2:
	sub	cx,2
	mov	es:[di],ax
	add	di,2
	jmp	faststosb_b0
faststosb_b3:
	dec	cx
	mov	es:[di],al
	inc	di
	jmp	faststosb_b0
ALIGN 16
faststosb_et LABEL WORD
	dw	OFFSET faststosb_e0
	dw	OFFSET faststosb_e1
	dw	OFFSET faststosb_e2
	dw	OFFSET faststosb_e3
faststosb_e1:
	mov	es:[di],al
	inc	di
	ret
faststosb_e3:
	mov	es:[di],ax
	mov	es:[di+2],al
	add	di,3
	ret
faststosb_e2:
	mov	es:[di],ax
	add	di,2
faststosb_e0:
	ret
ALIGN 16
faststosb PROC NEAR
	;fills CX bytes of data with AX to ES:DI aligning destination
	;di will point to the correct position after this (cx won't)
	cmp	cx,4
	jb	@@4
	mov	bx,di
	and	bx,3
	add	bx,bx
	jmp	faststosb_bt[bx]
faststosb_b0:
	mov	bx,ax
	shl	eax,16
	mov	ax,bx
	;eax=filler
	mov	bx,cx
	add	bx,bx
	and	bx,not 7
	neg	bx
	;bx=-cx*8
	add	bx,OFFSET @@jmp
	jmp	bx
	REPT	512
	db	66h,26h,89h,05h		;mov	es:[di],eax
	db	81h,0c7h,04h,00h	;add	di,0004h
	ENDM	;total 8 bytes
@@jmp:	mov	bx,cx
	and	bx,3
	add	bx,bx
	jmp	faststosb_et[bx]
@@4:	mov	bx,cx
	add	bx,bx
	jmp	faststosb_et[bx]
faststosb ENDP
*/
