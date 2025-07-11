#include <opencv2/opencv.hpp>
using namespace cv;
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <bits/stdc++.h>
using namespace std;
#include <cwctype>
#include <cmath>
#include <map>

int main(int argc, char *argv[])
{

    std::map<std::string, int> map_geneName_index;

    std::map<std::string, int> map_class_name;
    std::map<std::string, int> map_class_Index;
    std::map<int, std::string> map_index_geneName;
    std::multimap<std::string, std::string> map_gene_cat;

    std::ifstream datasetFile, lablesFile, inputfileMutualinfo, treeGnenMapfile, inputGenesCatfile, myfile;
    std::ofstream outputfile;
    std::ofstream outputfileClass;
    std::ofstream outputfilePatt;
    std::ofstream outputfileSelectedGenes;
    std::ofstream outputfileSelectedGenesExpression;
    std::ofstream outputfileSelectedGenesCat;

    std::string lineGene;
    //   string dataset = "3750_PBMC_data";  //1
    // string dataset = "Baron_HumanPancreas_data"; //2
    // string dataset = "DownSampled_SortedPBMC_data"; //3
    // string dataset = "Muraro_HumanPancreas_data"; // 4
    // string dataset = "Segerstolpe_HumanPancreas_data"; // 5
    // string dataset = "Xin_HumanPancreas_data"; // 6
    // string dataset = "Baron_MousePancreas_data"; // 7
    string dataset = "Merged_HumanPancreas_data"; // 8

    string tempFolder = "tempfolder/";
    string outputImagefolder = tempFolder + "outputImages/";
    //****************************************************
    string folderCreateCommand = "mkdir -p " + outputImagefolder;
    cout << "\n will apply cmd [" << folderCreateCommand << "]";
    system(folderCreateCommand.c_str());
    //*****************************************************

    string datasetfilename = "dataset/Filtered_" + dataset + ".csv";
    string labelsfilename = "dataset/Labels_" + dataset + ".csv";

    inputGenesCatfile.open("genes_3Levels_Cat.txt");
    datasetFile.open(datasetfilename);
    lablesFile.open(labelsfilename);

    outputfileSelectedGenes.open(tempFolder + "SelectedGenesinSorted.txt");
    outputfileSelectedGenesExpression.open(tempFolder + "SelectedGenesExpressionFeatuers2.csv");
    outputfileClass.open(tempFolder + "SelectedClasses.csv");
    outputfileSelectedGenesCat.open(tempFolder + "SelectedGenesNamesCat.csv");

    int uselog = 0;
    float geneExpressionFilter = 0;
    int topGeneCountThreshold = 500;
    float geneMaxFeatureValueThreshold = 5;
    int numperofIncludedGenesInImage = 1000;
    bool UseSortedGeneList = true;

    uselog = stoi(argv[1]);
    geneExpressionFilter = stof(argv[2]);
    topGeneCountThreshold = stoi(argv[3]);
    geneMaxFeatureValueThreshold = stof(argv[4]);
    numperofIncludedGenesInImage = stoi(argv[5]);
    int useDashSeparator = stoi(argv[6]);

    cout << " Use log " << uselog << " , Filter " << geneExpressionFilter << " Select top " << topGeneCountThreshold << " Sort genes for ML " << UseSortedGeneList;

    outputfileSelectedGenesCat << "L1,L2,L3,gname,AREA,hdi\n";
    while (std::getline(inputGenesCatfile, lineGene, '\n'))
    {
        int fx1 = lineGene.find("\t", 0);
        map_gene_cat.insert(pair<std::string, std::string>(lineGene.substr(0, fx1), lineGene.substr(fx1 + 1)));
    }
    cout << "\n Number of Genes => Cat is " << map_gene_cat.size();
    //****************************************************
    std::getline(datasetFile, lineGene, '\n');
    int fx2 = 0;
    int fx1 = 0;
    int gene_loop_Index = 0;
    do
    {
        fx1 = fx2;
        fx2 = lineGene.find(",", fx1 + 1);

        if (gene_loop_Index > 0 && fx2 > 0)
        {
            string gname = lineGene.substr(fx1 + 2, fx2 - fx1 - 3);
            int underScoreIndex = gname.find("_", 0);
            if (underScoreIndex > 0 && useDashSeparator == 1)
            {
                string gname1 = gname.substr(0, underScoreIndex);
                if (map_geneName_index.count(gname1) > 0)
                    gname1 = gname1.append("1");
                map_geneName_index[gname1] = gene_loop_Index - 1;
                map_index_geneName[gene_loop_Index - 1] = gname1;
            }
            else
            {
                map_geneName_index[gname] = gene_loop_Index - 1;
                map_index_geneName[gene_loop_Index - 1] = gname;
            }
        }
        if (fx2 == -1)
        {
            int fx3 = lineGene.find("\"", fx1 + 2);
            string gname = lineGene.substr(fx1 + 2, fx3 - fx1 - 2);
            int underScoreIndex = gname.find("_", 0);
            if (underScoreIndex > 0 && useDashSeparator == 1)
            {
                string gname1 = gname.substr(0, underScoreIndex);
                if (map_geneName_index.count(gname1) > 0)
                    gname1 = gname1.append("1");
                map_geneName_index[gname1] = gene_loop_Index - 1;
                map_index_geneName[gene_loop_Index - 1] = gname1;
            }
            else
            {
                map_geneName_index[gname] = gene_loop_Index - 1;
                map_index_geneName[gene_loop_Index - 1] = gname;
            }
        }

        gene_loop_Index++;
    } while (fx2 >= 0);

    int totalNumberofGenes = map_geneName_index.size();
    float maxgarray[totalNumberofGenes];
    for (int i = 0; i < map_geneName_index.size(); i++)
        maxgarray[i] = 0;

    float mingarray[totalNumberofGenes];
    for (int i = 0; i < map_geneName_index.size(); i++)
        mingarray[i] = 10000;

    float totgarray[totalNumberofGenes];
    for (int i = 0; i < map_geneName_index.size(); i++)
        totgarray[i] = 0;

    int countgarray[totalNumberofGenes];
    for (int i = 0; i < map_geneName_index.size(); i++)
        countgarray[i] = 0;

    float sumgarray[totalNumberofGenes];
    for (int i = 0; i < map_geneName_index.size(); i++)
        sumgarray[i] = 0;

    float tempsumgarray[totalNumberofGenes];
    for (int i = 0; i < map_geneName_index.size(); i++)
        tempsumgarray[i] = 0;
    //*****************************************************
    int numberofSamples = 0;
    while (std::getline(datasetFile, lineGene, '\n'))
    {
        numberofSamples++;
    }
    cout << "\n Input dataset: Number of samples " << numberofSamples << "\t Number of genes " << totalNumberofGenes;
    string *fullMatClasses = new string[numberofSamples];
    float **fullMatDataSet = new float *[numberofSamples];
    for (int i = 0; i < numberofSamples; i++)
        fullMatDataSet[i] = new float[totalNumberofGenes]();
    //*****************************************************
    std::getline(lablesFile, lineGene, '\n');
    for (int i = 0; i < numberofSamples; i++)
    {
        string classstr;
        std::getline(lablesFile, classstr, '\n');
        int lastCharIndex = 0;
        for (int i = 0; i < classstr.length(); i++)
            if (iswalnum(classstr[i]) == 0) // Checks if the given wide character is NOT an alphanumeric character
            {
                classstr[i] = '_';
            }
            else
            {
                lastCharIndex = i;
            }
        classstr = classstr.substr(1, lastCharIndex);
        fullMatClasses[i] = classstr;

        map_class_name[classstr]++;

        if (map_class_name[classstr] == 1)
        {
            map_class_Index[classstr] = map_class_Index.size();
        }
    }

    //****************************************************
    datasetFile.close();
    datasetFile.open(datasetfilename);
    std::getline(datasetFile, lineGene, '\n');
    int loopcounter = 0;

    cout << "\n Step 0: Buld fullMatDataSet \n";
    int part = numberofSamples / 10;
    while (std::getline(datasetFile, lineGene, '\n'))
    {
        int fx2 = lineGene.find(",");
        fx1 = 0;
        gene_loop_Index = 0;
        do
        {
            fx1 = fx2;
            fx2 = lineGene.find(",", fx1 + 1);
            float value = 0;
            if (fx2 > 0)
            {
                //cout <<"["<<lineGene.substr(fx1 + 1, fx2 - fx1 - 1)<<"]";
                //fflush(stdout);
                //getchar();
                value = stof(lineGene.substr(fx1 + 1, fx2 - fx1 - 1));
            }
            else // (fx2 == -1)
            {
                //cout <<"[" <<lineGene.substr(fx1 + 1)<<"]";
                //fflush(stdout);
                //getchar();
                value = stof(lineGene.substr(fx1 + 1));
            }

            if (value <= geneExpressionFilter) // filter out all genes with Expression value less than X
                value = 0;

            if (maxgarray[gene_loop_Index] < value)
                maxgarray[gene_loop_Index] = value;
            if (mingarray[gene_loop_Index] > value)
                mingarray[gene_loop_Index] = value;

            if (uselog)
                value = log2(1.0 + value);
            fullMatDataSet[loopcounter][gene_loop_Index] = value;
            gene_loop_Index++;

        } while (fx2 >= 0);
        loopcounter++;
        if (loopcounter % part == 0 || loopcounter == numberofSamples)
            cout << "[" << (100.0 * loopcounter) / numberofSamples << "%],";
        fflush(stdout);
    }

    cout << "\n Stage1 ";

    int outgenescount = 0;
    for (int i = 0; i < gene_loop_Index; i++)
    {
        if (maxgarray[i] < geneMaxFeatureValueThreshold || mingarray[i] == 10000)
        {
            outgenescount++;
            maxgarray[i] = 0;
        }
    }
    cout << "\n Count of not included genes " << outgenescount << " out of " << gene_loop_Index;
    cout << "\n Stage2: Normalize and Calc variation \n";
    fflush(stdout);
    part = numberofSamples / 10;
    loopcounter = 0;
    int numberofFilterValues = 0;
    for (int i = 0; i < numberofSamples; i++)
    {
        float maxv = 0;
        float tot = 0;
        float count = 0;

        for (int j = 0; j < totalNumberofGenes; j++)
            if (fullMatDataSet[i][j] > maxv)
                maxv = fullMatDataSet[i][j];

        for (int j = 0; j < totalNumberofGenes; j++)
        {
            if (maxv > 0)
                fullMatDataSet[i][j] = fullMatDataSet[i][j] / maxv;
            else
                fullMatDataSet[i][j] = 0;
            totgarray[j] += fullMatDataSet[i][j];
            countgarray[j]++;
        }

        loopcounter++;
        if (loopcounter % part == 0 || loopcounter == numberofSamples)
            cout << "[" << (100.0 * loopcounter) / numberofSamples << "%],";
        fflush(stdout);
    }

    loopcounter = 0;
    for (int i = 0; i < numberofSamples; i++)
    {
        for (int j = 0; j < totalNumberofGenes; j++)
        {
            if (countgarray[j] > 0 && maxgarray[j] > 0)
                sumgarray[j] += pow((fullMatDataSet[i][j] - (totgarray[j] / countgarray[j])), 2);
        }
    }

    for (int i = 0; i < totalNumberofGenes; i++)
    {
        if (countgarray[i] > 0 && maxgarray[i] > 0)
        {
            sumgarray[i] = sumgarray[i] / countgarray[i];
        }
        tempsumgarray[i] = sumgarray[i];
    }
    cout << "\n Sorting Var.. \n";
    fflush(stdout);
    gene_loop_Index = totalNumberofGenes;
    for (int i = 0; i < gene_loop_Index - 1; i++)
        for (int j = 0; j < gene_loop_Index - i - 1; j++)
        {
            if (tempsumgarray[j] < tempsumgarray[j + 1])
            {
                float temp = tempsumgarray[j + 1];
                tempsumgarray[j + 1] = tempsumgarray[j];
                tempsumgarray[j] = temp;
            }
        }

    cout << "\n max " << tempsumgarray[0] << " target " << tempsumgarray[topGeneCountThreshold];

    // Extract Images

    int *SelectedGenesOrdered = new int[topGeneCountThreshold]();
    int collectedGenes = 0;
    for (int gid = 0; gid < totalNumberofGenes; gid++)
    {
        if (sumgarray[gid] > tempsumgarray[topGeneCountThreshold] && collectedGenes < topGeneCountThreshold)
        {
            SelectedGenesOrdered[collectedGenes] = gid;
            collectedGenes++;
        }
    }

    if (UseSortedGeneList)
        for (int i = 0; i < topGeneCountThreshold - 1; i++)
            for (int j = 0; j < topGeneCountThreshold - i - 1; j++)
            {
                if (sumgarray[SelectedGenesOrdered[j]] < sumgarray[SelectedGenesOrdered[j + 1]])
                {
                    float temp = SelectedGenesOrdered[j + 1];
                    SelectedGenesOrdered[j + 1] = SelectedGenesOrdered[j];
                    SelectedGenesOrdered[j] = temp;
                }
            }

    std::getline(lablesFile, lineGene, '\n');
    loopcounter = 0;

    for (int i = 0; i < topGeneCountThreshold; i++)
    {
        outputfileSelectedGenes << map_index_geneName[SelectedGenesOrdered[i]] << "\t" << sumgarray[SelectedGenesOrdered[i]] << "\n";
    }
    outputfileSelectedGenes.close();
    //************* Generate The gene features
    bool firstWrite = true;
    for (int j = 0; j < topGeneCountThreshold; j++)
    {
        firstWrite ? outputfileSelectedGenesExpression << map_index_geneName[SelectedGenesOrdered[j]]
                   : outputfileSelectedGenesExpression << "," << map_index_geneName[SelectedGenesOrdered[j]];
        firstWrite = false;
    }
    outputfileSelectedGenesExpression << "\n";
    for (int i = 0; i < numberofSamples; i++)
    {
        bool firstWrite = true;
        for (int j = 0; j < topGeneCountThreshold; j++)
        {
            firstWrite ? outputfileSelectedGenesExpression << fullMatDataSet[i][SelectedGenesOrdered[j]]
                       : outputfileSelectedGenesExpression << "," << fullMatDataSet[i][SelectedGenesOrdered[j]];
            firstWrite = false;
        }
        outputfileSelectedGenesExpression << "\n";
    }
    cout << "\nfinish writing genes Features..";
    outputfileClass << "Class\n";
    for (int i = 0; i < numberofSamples; i++)
    {
        outputfileClass << map_class_Index[fullMatClasses[i]] << "\n";
    }
    outputfileClass.close();
    cout << "\nfinish writing class file";

    outputfileSelectedGenesExpression.close();

    // Call python to extract most significant genes

    folderCreateCommand = "python3 infoGain.py";
    cout << "\n will apply cmd [" << folderCreateCommand << "]";
    fflush(stdout);
    system(folderCreateCommand.c_str());
    inputfileMutualinfo.open(tempFolder + "mutual_info.csv");
    std::getline(inputfileMutualinfo, lineGene, '\n');
    fx2 = 0;
    int gcount = 0;
    for (int i = 0; i < topGeneCountThreshold; i++)
    {
        std::getline(inputfileMutualinfo, lineGene, '\n');
        fx2 = lineGene.find(",");
        string gname = lineGene.substr(0, fx2);
        SelectedGenesOrdered[gcount] = map_geneName_index[gname];
        gcount++;
    }
    outputfileSelectedGenesExpression.close();
    outputfileSelectedGenesExpression.open(tempFolder + "SelectedGenesExpressionFeatuers.csv");
    firstWrite = true;
    for (int j = 0; j < topGeneCountThreshold; j++)
    {
        firstWrite ? outputfileSelectedGenesExpression << map_index_geneName[SelectedGenesOrdered[j]]
                   : outputfileSelectedGenesExpression << "," << map_index_geneName[SelectedGenesOrdered[j]];
        firstWrite = false;
    }
    bool firstloopWrite = true;
    outputfileSelectedGenesExpression << "\n";
    for (int i = 0; i < numberofSamples; i = i + 1)
    {
        float maxv = 0;
        for (int j = 0; j < topGeneCountThreshold; j++)
        {
            float tempv = fullMatDataSet[i][SelectedGenesOrdered[j]];
            if (tempv > maxv)
                maxv = tempv;
        }

        bool firstWrite = true;
        int notIncluded = 0;
        for (int j = 0; j < topGeneCountThreshold; j++)
        {
            if (fullMatDataSet[i][SelectedGenesOrdered[j]] == 0)
            {
                firstWrite ? outputfileSelectedGenesExpression << "0"
                           : outputfileSelectedGenesExpression << ",0";
            }
            else
            {
                firstWrite ? outputfileSelectedGenesExpression << std::fixed << std::setprecision(3) << fullMatDataSet[i][SelectedGenesOrdered[j]] / maxv
                           : outputfileSelectedGenesExpression << "," << std::fixed << std::setprecision(3) << fullMatDataSet[i][SelectedGenesOrdered[j]] / maxv;
            }
            firstWrite = false;

            if (firstloopWrite && j < numperofIncludedGenesInImage)
            {
                int ver = 1;
                auto it = map_gene_cat.equal_range(map_index_geneName[SelectedGenesOrdered[j]]);
                // auto itr = it.first; // just one gene to single loc
                int itemsNum = std::distance(it.first, it.second);
                if (itemsNum > 0)
                {
                    for (auto itr = it.first; itr != it.second; ++itr)
                    {
                        if (itr->first == map_index_geneName[SelectedGenesOrdered[j]])
                        {
                            int fx1 = itr->second.find("_", 0);
                            int fx2 = itr->second.find("_", fx1 + 1);
                            int fx3 = itr->second.find("_", fx2 + 1);

                            // if (ver < 3)
                            outputfileSelectedGenesCat << itr->second.substr(0, 2) << ","
                                                       << itr->second.substr(fx1 + 1, 7) << ","
                                                       << itr->second.substr(fx2 + 1, fx3 - fx2 - 1) << ","
                                                       //<< itr->second.substr(fx2 + 1, fx3 - fx2 - 1) << ","
                                                       << itr->first << "_" << ver << ",2," << itr->second.substr(fx3 + 1, 6) << "\n";
                            ver++;
                        }
                    }
                }
                else
                {
                    notIncluded++;
                    outputfileSelectedGenesCat << "N4,ko01000,0," << map_index_geneName[SelectedGenesOrdered[j]] << "_1,2,1\n";
                }
            }
        }
        if (firstloopWrite)
            cout << "\nNot included in 3Levels = " << notIncluded;
        firstloopWrite = false;

        outputfileSelectedGenesExpression << "\n";
    }
    outputfileSelectedGenesExpression.close();
    outputfileSelectedGenesCat.close();
    cout << "\n Generate Tree Map Image";
    fflush(stdout);

    folderCreateCommand = "python3 buildTreeMapImage.py 100";
    cout << "\n will apply cmd [" << folderCreateCommand << "]";
    fflush(stdout);
    system(folderCreateCommand.c_str());
    myfile.open(tempFolder + "generect2.txt");
    vector<string> tokens;
    string lineCat;
    while (std::getline(myfile, lineCat, '\n'))
    {
        tokens.push_back(lineCat);
        //  cout << "\n"
        //       << lineGene;
        // fflush(stdout);
    }
    int numberofPixels = tokens.size() / 6;
    cout << " Number of Pixels " << numberofPixels;
    //getchar();
    // int down_width = pow(numberofPixels, 0.5) + 2;
    int down_width = 64;
    cout << "\ntoken from Rect : " << tokens.size();
    cout << " \n Extract Images \n";
    fflush(stdout);
    part = numberofSamples / 10;
    for (int sampleId = 0; sampleId < numberofSamples; sampleId += 1)
    {
        Mat imgx = imread("Image500.jpg", IMREAD_GRAYSCALE);
        Mat gimg;
        resize(imgx, gimg, Size(down_width, down_width), INTER_LINEAR_EXACT);
        float maxvinImage = 0;
        int checkImage[64][64];
        int errorcount = 0;
        std::fill(&checkImage[0][0], &checkImage[0][0] + 64 * 64, 0);
        for (int i = 0; i < tokens.size() / 6; i++)
        {
            string gname = tokens[i * 6 + 1];
            int x1 = -1;
            int x2 = -1;
            do
            {
                x2 = x1;
                x1 = gname.find("_", x1 + 1);
            } while (x1 > 0);
            string gname2 = gname.substr(0, x2);
            int gindex = map_geneName_index[gname2];
            if (fullMatDataSet[sampleId][gindex] > maxvinImage)
                maxvinImage = fullMatDataSet[sampleId][gindex];
        }

        for (int i = 0; i < tokens.size() / 6; i++)
        {
            string gname = tokens[i * 6 + 1];
            int x1 = -1;
            int x2 = -1;
            do
            {
                x2 = x1;
                x1 = gname.find("_", x1 + 1);
            } while (x1 > 0);
            string gname2 = gname.substr(0, x2);
            int gindex = map_geneName_index[gname2];
            unsigned char rr = 255 * (fullMatDataSet[sampleId][gindex] / maxvinImage);
            int x = round(0.01 * down_width * std::stof(tokens[i * 6 + 2]));
            int y = round(0.01 * down_width * std::stof(tokens[i * 6 + 3]));
            int w = round(0.01 * down_width * std::stof(tokens[i * 6 + 4]));
            int h = round(0.01 * down_width * std::stof(tokens[i * 6 + 5]));
            if (checkImage[x][y] == 0)
                checkImage[x][y] = 1;
            else
            {
                // cout << "\nerror " << sampleId << " " << x << " " << y << " " << errorcount;
                errorcount++;
                // getchar();
            }

            if (rr == 0)
                continue;
            cv::rectangle(gimg, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(rr), -1);
            // unsigned char &color = gimg.at<unsigned char>(x, y);
            // color = rr;
        }
        if (sampleId == 0)
            cout << "\n Number of errors " << errorcount;
        string sname = std::to_string(map_class_Index[fullMatClasses[sampleId]]);
        string lname = "_imIndex_px_";
        // string index = std::to_string(classes[classstr]);
        char buffer[8];
        sprintf(buffer, "%07d", sampleId);
        string type = ".bmp";
        string ff = outputImagefolder +
                    //+ sname + "/"
                    //+ sname +
                    buffer + type;
        imwrite(ff, gimg);
        loopcounter++;
        if (loopcounter % part == 0 || loopcounter == numberofSamples)
            cout << "[" << (100.0 * loopcounter) / numberofSamples << "%],";
        fflush(stdout);
    }
    cout << "\n";
    outputfileSelectedGenesExpression.close();
    outputfileSelectedGenes.close();
    return 0;
}
