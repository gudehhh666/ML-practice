#include<pic.h>
#include<string.h>

#define CS RC4
#define CLK RC3
#define SID RC5
#define RED RC1
#define GRE RC2
#define LINE1 1
#define LINE2 2

unsigned char bank3 W_COMMAND=0xF8;
unsigned char bank3 W_DATA=0xFA;

unsigned char bank3 password[8];
unsigned char bank3 pass_temp[5];
unsigned char bank3 room[4];

unsigned char bank2 LCD_IN[10]={0x00,0x10,0x20,0x00,0x00,0x20,0x00,0x60,0x00,0xC0};
unsigned char bank2 LCD_DDRAM1[2]={0x80,0x00};
unsigned char bank2 LCD_DDRAM2[2]={0x90,0x00};
unsigned char bank2 tip1[8]={0xCA,0xE4,0xC8,0xEB,0xBF,0xDA,0xC1,0xEE};
unsigned char bank2 tip2[10]={0xCA,0xE4,0xC8,0xEB,0xB7,0xBF,0xBC,0xE4,0xBA,0xC5};
unsigned char bank2 tip3[12]={0xD2,0xAA,0xD0,0xDE,0xB8,0xC4,0xBF,0xDA,0xC1,0xEE,0xC2,0xF0};
unsigned char bank2 tip4[10]={0xCA,0xE4,0xC8,0xEB,0xBE,0xC9,0xBF,0xDA,0xC1,0xEE};
unsigned char bank2 tip5[10]={0xCA,0xE4,0xC8,0xEB,0xD0,0xC2,0xBF,0xDA,0xC1,0xEE};
unsigned char bank2 tip6[2]={0xA3,0xB0};
unsigned char bank2 G[5];

unsigned char bank1 LCD_PASS[8]={0xBB,0xB6,0xD3,0xAD,0xBB,0xD8,0xC0,0xB4};
unsigned char bank1 LCD_WARNING[8]={0xBF,0xDA,0xC1,0xEE,0xB4,0xED,0xCE,0xF3};
unsigned char bank1 LCD_FORBID[8]={0xBD,0xFB,0xD6,0xB9,0xCD,0xA8,0xB9,0xFD};

unsigned char key;
unsigned char key_temp;
unsigned char spy;
unsigned char keyfind(void);
unsigned char keyscan(void);
unsigned char H;

void delay(unsigned char a);

void LCD_INITIAL(void);
void LCD_CLEAR(void);
void LCD_START(char line,char first);
void LCD_WRITE(unsigned char data);

void sci_initial(void);
unsigned char sci_rc(void);
void sci_tx(unsigned char bb);

void local(void);
void visit(void);
void modify_key(void);
void set_rs232(void);
void clock();

unsigned char compare(void);
unsigned char calibration(void);

void main()
{
    TRISD=0xF0;
	PORTC=0x00;
	TRISC=0xC0;
	SSPSTAT=0x40;
	SSPCON=0x20;
	CS=1;
	LCD_CLEAR();
	LCD_INITIAL();
	sci_initial();
	set_rs232();
	LCD_CLEAR();
	while(1)
	{
		char i;
		LCD_INITIAL();
		LCD_START(LINE2,0);
		key=keyfind();
		if(key==10)
		{
			LCD_INITIAL();
			LCD_START(LINE1,0);
			for(i=0;i<8;i++)
			LCD_WRITE(tip1[i]);
			local();
		}
		else if(key==11)
		{
			LCD_CLEAR();
			LCD_INITIAL();
			LCD_START(LINE1,1);
			for(i=0;i<10;i++)
			LCD_WRITE(tip2[i]);
			visit();
		}
		else if(key==13)
			modify_key();
		delay(50);
		RED=0;
		GRE=0;
		LCD_INITIAL();
	}
}

void set_rs232(void)
{
	char i;
	for(i=0;i<8;i++)
	{
		password[i]=sci_rc();
		delay(2);
	}
}

unsigned char calibration(void)
{
	char i;
	unsigned char key_temp;
	for(i=0;i<6;i++)
	{
		key_temp=keyfind();
		if(key_temp==12)
		return 2;
		else if(key_temp==14)
		{
			i=0;
			continue;
		}
		else if((key_temp==15)&(i==5))
		break;
		pass_temp[i]=key_temp;
	}
	if(compare()==1)
	return 1;
	else 
	return 0;
}

unsigned char compare(void)
{
	char i;
	for(i=0;i<8;i++)
	{
		if(pass_temp[i]!=password[i])
		return 0;
	}
	return 1;
}

void local(void)
{
	char i,cnt=0;
	while(cnt<3)
	{  
		if(calibration()==1)
		{
			LCD_INITIAL();
			LCD_START(LINE1,2);
			for(i=0;i<4;i++)
			LCD_WRITE(LCD_PASS[i]);
			GRE=1;
			RED=0;
			break;
		}
		else if(calibration()==0)
		{
			LCD_INITIAL();
			LCD_START(LINE1,2);
			for(i=0;i<8;i++)
			LCD_WRITE(LCD_WARNING[i]); 
			GRE=1;
			RED=1;
			cnt++;
		}
		else if(calibration()==2)
		return;
		if(cnt==3)
		{
			LCD_INITIAL();
			LCD_START(LINE1,2);
			for(i=0;i<8;i++)
			LCD_WRITE(LCD_FORBID[i]);
			GRE=0;
			RED=1;
			break;
		}
	}
}

void visit(void)
{
	unsigned char room[4],key_temp; 
	char i;
	for(i=0;i<5;i++)
	{	
		key_temp=keyfind();
		if(key_temp==12)
		return;
		else if(key_temp==14)
		{
			i=0;
			continue;
		}
		else if((key_temp==15)&(i==4))
		break;
		room[i]=key_temp;
	}
	LCD_INITIAL();
	LCD_START(LINE1,1);
	H=0;
	for(i=0;i<4;i++)
	{
		LCD_WRITE(tip6[0]);
		H=room[i];
        G[i]=tip6[1]+H;
		LCD_WRITE(G[i]);
	}
	for(i=0;i<4;i++)
	sci_tx(room[i]);
	if(sci_rc()==1)
	{
		LCD_INITIAL();
		LCD_START(LINE1,2);
		for(i=0;i<4;i++)
		LCD_WRITE(LCD_PASS[i]);
		RED=0;
		GRE=1;
	}
	else
	{
		LCD_INITIAL();
		LCD_START(LINE1,2);
		for(i=0;i<8;i++)
		LCD_WRITE(LCD_FORBID[i]);
		RED=1;
		GRE=0;
	}
}

void modify_key(void)
{
	char i,j,cnt=0;
	LCD_INITIAL();
	LCD_START(LINE1,1);
	for(i=0;i<12;i++)
	LCD_WRITE(tip3[i]);
	if(keyfind()==15)
	{
		LCD_INITIAL();
		LCD_START(LINE1,1);
		for(i=0;i<10;i++)
		LCD_WRITE(tip4[i]);
		while(cnt<5)
		{
			if(calibration()==1)
			{
				LCD_INITIAL();
				LCD_START(LINE1,1); 
				for(i=0;i<10;i++)
				LCD_WRITE(tip5[i]);
				for(i=0;i<9;i++)
				{
					key_temp=keyfind();
					if(key_temp==12)
					return;
					else if(key_temp==14)
					{
						i=0;
						continue;
					}
					else if((key_temp==15)&(i==8))
					break;
					pass_temp[i]=key_temp;
				}
				for(j=0;j<8;j++)
				{
					password[j]=pass_temp[j];
				}
				LCD_INITIAL();
				LCD_START(LINE1,1); 
				for(i=0;i<8;i++)
				LCD_WRITE(LCD_PASS[i]);
				break;
			}
			else
			{
				LCD_INITIAL();
				LCD_START(LINE1,1);
				for(i=0;i<10;i++)
				LCD_WRITE(tip4[i]);
				cnt++;
				GRE=1;
				RED=1;
			}
			if(cnt==4)
			return;
		}
	}
	else return;
}

void LCD_INITIAL()
{   
	unsigned char n=0;
	for(n;n<10;n++)
	{
		SSPBUF=W_COMMAND;
		while(!SSPIF);
		SSPIF=0x00;
		SSPBUF=LCD_IN[n];
		while(!SSPIF);
		SSPIF=0x00;
		n++;
		SSPBUF=LCD_IN[n];
		while(!SSPIF);
		SSPIF=0x00;
		delay(1);
	}
}

void  LCD_CLEAR(void)
{
	SSPBUF = W_COMMAND;
	while(!SSPIF);
	SSPIF=0x00;
	SSPBUF=0x00;
	while(!SSPIF);
	SSPIF=0x00;
	SSPBUF=0x10;
	while(!SSPIF);
	SSPIF=0x00;
}
void LCD_START(char line,char first)
{
	unsigned char n;
	unsigned char i;
	if(line==LINE1)
	{
		SSPBUF=W_COMMAND;
		while(!SSPIF);
		SSPIF=0x00;
		SSPBUF=LCD_DDRAM1[0];
		while(!SSPIF);                             
		SSPIF=0x00;                                     
		SSPBUF=(LCD_DDRAM1[1]+first)<<4;
		while(!SSPIF);                           
		SSPIF=0x00;                                       
	}
	else
	{
		SSPBUF=W_COMMAND;
		while(!SSPIF);
		SSPIF=0x00;
		SSPBUF=LCD_DDRAM2[0];
		while(!SSPIF);
		SSPIF=0x00;
		SSPBUF=(LCD_DDRAM2[1]+first)<<4;
		while(!SSPIF);   
	}                  
	SSPIF=0x00;
}      
                        
void LCD_WRITE(unsigned char data)
{
	SSPBUF=W_DATA;
	while(!SSPIF);
	SSPIF=0X00;
	SSPBUF=data&0xF0;
	while(!SSPIF);
	SSPIF=0X00;
	SSPBUF=(data&0x0F)<<4;
	while(!SSPIF);
	SSPIF=0x00;
	delay(1);
}

void sci_initial()
{
	SYNC=0;
	TX9=0;
	TXEN=0;
	BRGH=0;
	SPBRG=77;
	RCSTA=0X90;
	TXIE=0;
	RCIE=0;
}

unsigned char sci_rc(void)
{
	unsigned char x;
	while(!RCIF);
	RCREG=0x00;
	RCIF=0;
	x=RCREG-0x30;
	delay(2);
	return(x);
}

void sci_tx(unsigned char bb)
{
	TXEN=1;
	TXREG=bb;
	while(!TXIF);	
	TXIF=0;
	delay(1);
	TXEN=0;
}


//
unsigned char keyfind(void)
{
	unsigned char j,temp1;
	unsigned char temp2=0x80;
	PORTD=0xF0;
	while(1)
	{
 		if(PORTD!=0xF0)
		{
			delay(5);
			if(PORTD!=0xF0)
			break;
		}
	}
	for(j=0;j<4;j++)
	{
		temp1=PORTD;
		temp1=temp1&temp2;    //1101 1101
		if(temp1==0)          //0010 0000
		break;
		temp2=temp2>>1;
	}
	return(keyscan()+j);     // j = 2
}


//i +j = 15  
unsigned char keyscan(void)
{
	unsigned char scancode=0xF7,i,a,b;
	unsigned char temp;
	for(i=0;i<13;)
	{
        //
		PORTD=scancode;
		temp=PORTD; 
        //1101 1101 

		temp=temp&0xF0;


        //
		if(temp!= (0x0F & scancode))
		    break;
        /*
        1111 0111
        --->
        1111 1011
        --->
        1111 1101
        1111 1110

        */
		a=scancode>>1;
		b=scancode<<7;
		scancode=a|b;
		//
        i=i+4;  //  i = 8
	}
	return(i);
}
//

void delay(unsigned char a)
{
	unsigned  int  i,j;
	for(i=0;i<a;i++)
	{
		for(j=0;j<5000;j++);
	}
}
