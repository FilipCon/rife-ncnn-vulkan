#include "filesystem_utils.h"
#include "rife.h"

// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int loadImage(const std::string& imagepath, ncnn::Mat& image) {
    unsigned char* pixeldata = 0;
    int w;
    int h;
    int c;
    FILE* fp = fopen(imagepath.c_str(), "rb");
    if (fp) {
        // read whole file
        unsigned char* filedata = 0;
        int length = 0;
        {
            fseek(fp, 0, SEEK_END);
            length = ftell(fp);
            rewind(fp);
            filedata = (unsigned char*) malloc(length);
            if (filedata) { fread(filedata, 1, length, fp); }
            fclose(fp);
        }

        if (filedata) {
            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
            c = 3;
            free(filedata);
        }
    }

    if (!pixeldata) { return -1; }

    image = ncnn::Mat(w, h, (void*) pixeldata, (size_t) 3, 3);
    return 0;
}

static int saveImage(const std::string& imagepath, const ncnn::Mat& image) {
    int success = 0;

    std::string ext = get_file_extension(imagepath);

    if (ext == PATHSTR("png") || ext == PATHSTR("PNG")) {
        success = stbi_write_png(imagepath.c_str(), image.w, image.h,
                                 image.elempack, image.data, 0);

    } else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") ||
               ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")) {
        success = stbi_write_jpg(imagepath.c_str(), image.w, image.h,
                                 image.elempack, image.data, 100);
    }

    if (!success) {
        fprintf(stderr, "encode image %s failed\n", imagepath.c_str());
    }

    return success ? 0 : -1;
}
