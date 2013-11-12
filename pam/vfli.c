#include <stdio.h>
#include <stdio.h>
#include <string.h>
//#include <dos.h>

//unsigned char far *vram=(char far *)0xa0000000;
unsigned char vram[320 * 200];
unsigned char fr[320 * 200 * 3];

int	blocksleft=999;

int	wid,hig,pics;

long	cnt[256];
char	lookup[256];

char	palette[768];
char	ownpal[768];

int	docoltable;
int	colorlookup;
int	oktoviewpal;
int	useownpal;
int	savefcp;

FILE	*out;

static void write_frame(unsigned char *buf, int size, int num, FILE *f)
{
	static int n = 0;
	char name[80], cmd[200];
	FILE *ff;
	int i;

	for (i = 0; i < 64000; i++) {
		int p = buf[i];
		fr[i * 3] = palette[p * 3] << 2;
		fr[i * 3 + 1] = palette[p * 3 + 1] << 2;
		fr[i * 3 + 2] = palette[p * 3 + 2] << 2;
	}
	
	snprintf(name, 80, "frame_%03d", n++);
	ff = fopen(name, "wb");
	fwrite(fr, size * 3, num, ff);
	fclose(ff);

	snprintf(cmd, 200, "convert -size 320x200 -depth 8 rgb:%s %s.png",
								name, name);
	system(cmd);
	unlink(name);
}

main(int argc,char *argv[])
{
	int	a;
	FILE	*f1;
	FILE	*ppp;
	f1=fopen(argv[1],"rb");
	if(f1==NULL) return(0);
	out=fopen("out.u","wb");
	if(out==NULL) return(0);

	//_asm	mov	ax,19
	//_asm	int	10h

	for(a=0;a<256;a++) lookup[a]=a;

	docoltable=1;
	oktoviewpal=0;
	useownpal=0;
	savefcp=1;
	rewind(f1); readflic(f1,0);

#if 0
	ppp=fopen("pal.pal","wb");
	if(ppp!=0)
		{
		fwrite(palette,1,768,ppp);
		fclose(ppp);
		}
#endif

	//_asm	mov	ax,3
	//_asm	int	10h
	fclose(f1);
}

// DOS-compatible dos_getw()
static int dos_getw(FILE *f)
{
	int val;

	val = getc(f);
	val += 256 * getc(f);

	return val;
}

int	readflic(FILE *f1,int flag)
{
	int	a;
	long	len,l,ll;
	len=dos_getw(f1);
	len+=(long)dos_getw(f1)<<16;
	dos_getw(f1);
	pics=dos_getw(f1);
	wid=dos_getw(f1);
	hig=dos_getw(f1);
	fseek(f1,0x80L,SEEK_SET);
	/*printf("\n\n\n\n\n\n\n\n\n\n\n");*/
	for(;;)
	{
		l=ftell(f1);
		l+=(long)dos_getw(f1);
		l+=(long)dos_getw(f1)<<16;
		a=dos_getw(f1);
		if(feof(f1)) break;
		if(a==0xf1fa)
		{
			int	a;
			blocksleft=dos_getw(f1);
			for(a=0;a<8;a++) getc(f1);
			ll=l;
		}
		else
		{
			doblock(f1,(unsigned)a);
			fseek(f1,l,SEEK_SET);
			blocksleft--;
			if(blocksleft==0)
			{
				fseek(f1,ll,SEEK_SET);
				{
					int	x,y;
					unsigned u;
					for(y=0;y<hig;y++)
					{
						u=y*320;
						for(x=0;x<wid;x++) cnt[vram[u++]]++;
					}
				}
				if(savefcp)
				{
					static int framecnt=0;
					int	x,y;
					{
//						outportb(0x3c8,7);
//						outportb(0x3c9,47);
//						outportb(0x3c9,47);
//						outportb(0x3c9,47);
						write_frame(vram,1,64000,out);
					}
				}
				//if(flag&1) getch();
			}
		}
	}
}

int	viewpal(void)
{
#if 0
	int	a;
	unsigned u;
	for(a=0;a<256;a++)
	{
		u=(a>>4)*3+320-48+((a&15)*3+200-48)*320;
		vram[u]=a;
		vram[u+1]=a;
		vram[u+2]=a;
		vram[320+u]=a;
		vram[320+u+1]=a;
		vram[320+u+2]=a;
		vram[640+u]=a;
		vram[640+u+1]=a;
		vram[640+u+2]=a;
	}
#endif
}

int	doblock(FILE *f1,unsigned type)
{
	if((type==0x000b || type==0x0004) && useownpal)
	{
		int	a;
		//outportb(0x3c8,0);
		//for(a=0;a<768;a++) outportb(0x3c9,ownpal[a]);
		if(oktoviewpal) viewpal();
		return(0);
	}
	if(type==0x000b)
	{
		int	a,c,d;
		c=getc(f1)*256; /* count */
		c+=getc(f1);
		c*=3;
		a=getc(f1)*256; /* first */
		a+=getc(f1);
		//outportb(0x3c8,a);
		//for(a=0;a<c;a++) outportb(0x3c9,palette[a]=getc(f1));
		for(a=0;a<c;a++) palette[a]=getc(f1);
		if(oktoviewpal) viewpal();
	}
	else if(type==0x0004)
	{
		int	a,c,d;
		c=getc(f1)*256; /* count */
		c+=getc(f1);
		c*=3;
		a=getc(f1)*256; /* first */
		a+=getc(f1);
		//outportb(0x3c8,a);
		//for(a=0;a<c;a++) outportb(0x3c9,palette[a]=(getc(f1)>>2));
		for(a=0;a<c;a++) palette[a]=(getc(f1)>>2);
		if(oktoviewpal) viewpal();
	}
	else if(type==0x000c)
	{
		int	a,b,c,lc,cc,cm,lb,le,lad;
		long	l;
		unsigned int u;
		lb=dos_getw(f1);
		le=lb+dos_getw(f1);
		for(lc=lb;lc<le;lc++)
		{
			cm=getc(f1);
			u=lc*320;
			for(cc=0;cc<cm && u<64000;cc++)
			{
				a=getc(f1);
				u+=a;
				a=getc(f1);
				if(a<0x80)
				{
					for(c=0;c<a;c++) vram[u++]=lookup[getc(f1)];
				}
				else
				{
					a=256-a;
					b=lookup[getc(f1)];
					for(c=0;c<a;c++) vram[u++]=b;
				}
			}
		}
		u=0;
	}
	else if(type==0x0007)
	{
		int	a,b,b1,b2,c,lc,cc,cm,lb,le,lad;
		int	skipcnt;
		long	l;
		unsigned int u;
		le=dos_getw(f1);
		for(lc=lad=0;lc<le;lc++)
		{
			cm=dos_getw(f1);
			while(cm<0)
			{
				lad=320*-cm;
				cm=dos_getw(f1);
			}
			u=lc*320+lad;
			for(cc=0;cc<cm && u<64000;cc++)
			{
				a=getc(f1);
				u+=a;
				a=getc(f1);
				if(a<0x80)
				{
					for(c=0;c<a;c++) 
					{
						vram[u++]=lookup[getc(f1)];
						vram[u++]=lookup[getc(f1)];
					}
				}
				else
				{
					a=256-a;
					b1=lookup[getc(f1)];
					b2=lookup[getc(f1)];
					for(c=0;c<a;c++)
					{
						vram[u++]=b1;
						vram[u++]=b2;
					}
				}
			}
		}
		u=0;
	}
	else if(type==0x000f)
	{
		int	a,b,c,lc,cc,cm;
		long	l;
		unsigned int u;
		for(lc=0;lc<hig;lc++)
		{
			u=lc*320;
			cm=getc(f1);
			for(cc=0;cc<cm && u<64000;cc++)
			{
				a=getc(f1);
				if(a<0x80)
				{
					b=lookup[getc(f1)];
					for(c=0;c<a;c++) vram[u++]=b;
				}
				else
				{
					a=256-a;
					for(c=0;c<a;c++) vram[u++]=lookup[getc(f1)];
				}
			}
		}
		u=0;
	}
	else
	{
		printf("UNKNOWN BLOCK TYPE: %04X\n",type);
		//getch();
	}
}
