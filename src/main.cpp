#include "dds.h"

#include <iostream>
#include <filesystem>

void Help() 
{

}

int main(int argc, char* argv[])
{
    if (argc < 2) 
    {
        return 1;
    }

    DdsImage img;
    if (!LoadDDS(argv[1], img)) 
    {
        printf("Load failed: %s\n", argv[1]);
    }

    printf("OK: %ux%u, mips=%u, format=%d\n", img.width, img.height, img.mip_count, (int)img.format);
    return 0;
}