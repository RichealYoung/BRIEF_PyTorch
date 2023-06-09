#include <iostream>
#include <math.h>
#include "tiffio.h"
#include <vector>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <ctime>
using namespace std;

uint32_t D, H, W;

float alpha(float x)
{
    return 0.8 * (pow(2, x / 6) - 1);
}

float beta(float x)
{
    return 0.5 * x - 7;
}

float clip(float x, float min, float max)
{
    if (x < min)
        x = min;
    if (x > max)
        x = max;
    return x;
}

int judge_filter(int p1, int p0, int q0, int q1, int index_a, int index_b, int thres)
{
    if ((p1 + p0 + q0 + q1) / 4 > thres)
        return 0; // The block artifacts of high brightness area is not obvious
    if (abs(p0 - q0) < alpha(index_a) && abs(p1 - p0) < beta(index_b) && abs(q1 - q0) < beta(index_b))
        return 1;
    else
        return 0;
}

std::vector<uint16_t> filter(uint16_t p2, uint16_t p1, uint16_t p0, uint16_t q0, uint16_t q1, uint16_t q2, uint16_t index_b)
{
    std::vector<uint16_t> result;
    float delta0, deltap1, deltaq1;
    uint16_t c0, c1;
    // basic filter operation
    delta0 = (4 * (q0 - p0) + (p1 - q1) + 4) / 8;
    deltap1 = (p2 + (p0 + q0 + 1) / 2 - 2 * p1) / 2;
    deltaq1 = (q2 + (q0 + p0 + 1) / 2 - 2 * q1) / 2;
    // clipping
    c1 = 20;
    c0 = c1;
    if (abs(p2 - p0) < beta(index_b)) // luminance
        c0 += 1;
    if (abs(q2 - q0) < beta(index_b)) // luminance
        c0 += 1;
    delta0 = clip(delta0, -c0, c0);
    deltap1 = clip(deltap1, -c1, c1);
    deltaq1 = clip(deltaq1, -c1, c1);
    // result
    p1 += deltap1;
    p0 += delta0;
    q0 -= delta0;
    q1 += deltaq1;
    result.push_back((uint16_t)p1);
    result.push_back((uint16_t)p0);
    result.push_back((uint16_t)q0);
    result.push_back((uint16_t)q1);
    return result;
}

std::vector<string> split(string str, string separator)
{
    std::vector<string> result;
    int cutAt;
    while ((cutAt = str.find_first_of(separator)) != str.npos)
    {
        if (cutAt > 0)
        {
            result.push_back(str.substr(0, cutAt));
        }
        str = str.substr(cutAt + 1);
    }
    if (str.length() > 0)
    {
        result.push_back(str);
    }
    return result;
}

void printVector(vector<string> vector1)
{
    for (int i = 0; i < int(vector1.size()); i++)
    {
        cout << vector1[i] << endl;
    }
}

void printVector(vector<int> vector1)
{
    for (int i = 0; i < int(vector1.size()); i++)
    {
        cout << vector1[i] << endl;
    }
}

std::vector<uint16_t> load_3d_data_uint16(const std::string &filename, uint32_t &depth, uint32_t &height, uint32_t &width)
{
    TIFF *tif = TIFFOpen(filename.c_str(), "r");
    if (tif == nullptr)
    {
        std::cerr << "Wrong data path !" << std::endl;
        abort();
    }
    uint16_t bitdpeh;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitdpeh);
    if (bitdpeh != 16)
    {
        std::cerr << "Only support 16 bit depth data !" << std::endl;
        abort();
    }
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    depth = TIFFNumberOfDirectories(tif);
    printf("Loading target data from %s\n", filename.c_str());
    printf("Target data shape (%d,%d,%d)\n", depth, height, width);

    std::vector<uint16_t> data(depth * height * width);
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            TIFFReadScanline(tif, &data[d * height * width + h * width], h);
        }
        TIFFReadDirectory(tif);
    }
    TIFFClose(tif);
    return data;
}

void save_3d_data_uint16(const std::string filename, std::vector<uint16_t> data, uint32_t depth, uint32_t height, uint32_t width)
{
    TIFF *out = TIFFOpen(filename.c_str(), "w");
    if (out)
    {
        int d = 0;
        do
        {
            TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
            TIFFSetField(out, TIFFTAG_PAGENUMBER, depth);
            TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
            TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
            TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
            TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, height);
            for (int h = 0; h < height; h++)
            {
                TIFFWriteScanline(out, &data[d * height * width + h * width], h, 0);
            }
            d++;
        } while (TIFFWriteDirectory(out) && d < depth);
        TIFFClose(out);
    }
    printf("Saving target data to %s\n", filename.c_str());
}

std::vector<string> listdir(const std::string &directoryPath)
{
    DIR *dir;
    std::vector<string> results;
    struct dirent *entry;
    dir = opendir(directoryPath.c_str());
    if (!dir)
    {
        std::cerr << "Failed to open directory: " << directoryPath << std::endl;
    }
    else
    {
        while ((entry = readdir(dir)) != nullptr)
        {
            std::string filename = entry->d_name;
            if (filename == "." || filename == "..")
                continue;
            results.push_back(filename);
        }
        closedir(dir);
    }
    return results;
}

void get_size(const std::string &filename)
{
    // ct1-0_256-0_256-0_256_decompressed.tif
    std::vector<string> split_info = split(filename, "-");
    std::string d = split_info[1], h = split_info[2], w = split_info[3];
    std::string d1 = split(d, "_")[0], d2 = split(d, "_")[1];
    std::string h1 = split(h, "_")[0], h2 = split(h, "_")[1];
    std::string w1 = split(w, "_")[0], w2 = split(w, "_")[1];
    std::vector<uint32_t> size;
    D = atoi(d2.c_str()) - atoi(d1.c_str());
    H = atoi(h2.c_str()) - atoi(h1.c_str());
    W = atoi(w2.c_str()) - atoi(w1.c_str());
}

struct LINE
{
    uint16_t z, l, r, d, u; // depth, left, right, down, up
};

bool judge_same(LINE line1, LINE line2)
{
    if (line1.z == line2.z && line1.l == line2.l && line1.r == line2.r && line1.d == line2.d && line1.u == line2.u)
        return true;
    else
        return false;
}

struct LINE line_value(struct LINE line, uint16_t z, uint16_t l, uint16_t r, uint16_t d, uint16_t u)
{
    line.z = z, line.l = l, line.r = r, line.d = d, line.u = u;
    return line;
}

void deblock(std::string step_dir, uint32_t index_a, uint32_t index_b, uint32_t thres)
{
    // load img
    std::string decompressed_dir = step_dir + "/decompressed";
    std::vector<string> dirs = listdir(decompressed_dir);
    std::string origin_name = dirs[0];
    std::string save_name = origin_name.substr(0, origin_name.size() - 4) + "_deblocked_c++.tif";
    std::string img_path = decompressed_dir + "/" + origin_name;
    std::string save_dir = step_dir + "/deblock";
    std::string commad = "mkdir -p " + save_dir;
    system(commad.c_str());
    std::string save_path = save_dir + "/" + save_name;
    std::string module_dir = step_dir + "/compressed/module";
    get_size(origin_name);
    std::vector<uint16_t> img = load_3d_data_uint16(img_path, D, H, W);
    printf("index_a:%d,index_b:%d,thres:%d\n", index_a, index_b, thres);
    std::vector<string> block_infos = listdir(module_dir);
    // find lines
    std::vector<LINE> lines;
    struct LINE lline, rline, dline, uline, line;
    bool lflag = false, rflag = false, dflag = false, uflag = false;
    for (uint16_t i = 0; i < block_infos.size(); i++)
    {
        std::vector<string> dhw = split(block_infos[i], "-");
        uint16_t z1 = atoi(split(dhw[0], "_")[1].c_str()), z2 = atoi(split(dhw[0], "_")[2].c_str());
        uint16_t y1 = atoi(split(dhw[1], "_")[1].c_str()), y2 = atoi(split(dhw[1], "_")[2].c_str());
        uint16_t x1 = atoi(split(dhw[2], "_")[1].c_str()), x2 = atoi(split(dhw[2], "_")[2].c_str());
        for (uint16_t i = 0; i < lines.size(); i++)
        {
            if (judge_same(lines[i], line_value(lline, z1, x1, x1, y1, y2)))
                lflag = true;
            if (judge_same(lines[i], line_value(rline, z1, x2, x2, y1, y2)))
                rflag = true;
            if (judge_same(lines[i], line_value(dline, z1, x1, x2, y1, y1)))
                dflag = true;
            if (judge_same(lines[i], line_value(uline, z1, x1, x2, y2, y2)))
                uflag = true;
        }
        for (uint16_t z = z1; z < z2 + 1; z++)
        {
            if (!lflag)
                lines.push_back(line_value(line, z, x1, x1, y1, y2));
            if (!rflag)
                lines.push_back(line_value(line, z, x2, x2, y1, y2));
            if (!dflag)
                lines.push_back(line_value(line, z, x1, x2, y1, y1));
            if (!uflag)
                lines.push_back(line_value(line, z, x1, x2, y2, y2));
        }
    }
    // deblock
    uint16_t p2, p1, p0, q0, q1, q2;
    vector<uint16_t> results;
    for (uint16_t i = 0; i < lines.size(); i++)
    {
        line = lines[i];
        clock_t startTime1 = clock();
        if ((line.l == line.r) && (line.l - 3 < 0 || line.l + 3 > W - 1))
            continue;
        else if ((line.d == line.u) && (line.d - 3 < 0 || line.d + 3 > H - 1))
            continue;
        for (uint16_t y = line.d; y < line.u + 1; y++)
        {
            for (uint16_t x = line.l; x < line.r + 1; x++)
            {
                if (line.l == line.r)
                {
                    p2 = img[line.z * H * W + y * W + x - 3], p1 = img[line.z * H * W + y * W + x - 2], p0 = img[line.z * H * W + y * W + x - 1];
                    q0 = img[line.z * H * W + y * W + x + 0], q1 = img[line.z * H * W + y * W + x + 1], q2 = img[line.z * H * W + y * W + x + 2];
                    if (judge_filter(p1, p0, q0, q1, index_a, index_b, thres))
                    {
                        results = filter(p2, p1, p0, q0, q1, q2, index_b);
                        img[line.z * H * W + y * W + x - 2] = results[0];
                        img[line.z * H * W + y * W + x - 1] = results[1];
                        img[line.z * H * W + y * W + x + 0] = results[2];
                        img[line.z * H * W + y * W + x + 1] = results[3];
                    }
                }
                else if (line.d == line.u)
                {
                    p2 = img[line.z * H * W + (y - 3) * W + x], p1 = img[line.z * H * W + (y - 2) * W + x], p0 = img[line.z * H * W + (y - 1) * W + x];
                    q0 = img[line.z * H * W + (y + 0) * W + x], q1 = img[line.z * H * W + (y + 1) * W + x], q2 = img[line.z * H * W + (y + 2) * W + x];
                    if (judge_filter(p1, p0, q0, q1, index_a, index_b, thres))
                    {
                        results = filter(p2, p1, p0, q0, q1, q2, index_b);
                        img[line.z * H * W + (y - 2) * W + x] = results[0];
                        img[line.z * H * W + (y - 1) * W + x] = results[1];
                        img[line.z * H * W + (y + 0) * W + x] = results[2];
                        img[line.z * H * W + (y + 1) * W + x] = results[3];
                    }
                }
            }
        }
    }
    save_3d_data_uint16(save_path, img, D, H, W);
}

// g++ deblock.cpp -o deblock -ltiff
int main(int argc, char **argv)
{
    std::string step_dir = argv[1];
    uint32_t index_a = 51, index_b = 2000, thres = 65535;
    deblock(step_dir, index_a, index_b, thres);
}
