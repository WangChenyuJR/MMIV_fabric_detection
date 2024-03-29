/*****************************************************************************
 * File generated by HDevelop Version 20.11
 *
 * Do not modify!
 *****************************************************************************/

#ifndef MAIN_H
#define MAIN_H

#ifndef __APPLE__
  #include <halconcpp/HalconCpp.h>
#else
  #include <HALCONCpp/HalconCpp.h>
#endif

namespace MMIV {

/*****************************************************************************
 * SetResourcePath
 *****************************************************************************
 * Use SetResourcePath in your application to specify the location of the 
 * HDevelop script or procedure library.
 *****************************************************************************/
  void SetResourcePath(const char* resource_path);
#ifdef _WIN32
  void SetResourcePath(const wchar_t* resource_path);
#endif

  void borda_count(
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& ScoreMatrix,
    HalconCpp::HTuple const& ExteriorPointRate,
    HalconCpp::HTuple* BordaScoreList,
    HalconCpp::HTuple* MaxNumIndexSorted);


  void cal_common_divisor(
    HalconCpp::HTuple const& input1,
    HalconCpp::HTuple const& input2,
    HalconCpp::HTuple* common_divisor);


  void cal_imgcrop_gray_mean(
    HalconCpp::HObject const& ImageObj_cropped,
    HalconCpp::HTuple const& gray_max_val,
    HalconCpp::HTuple* Gray_Mean,
    HalconCpp::HTuple* gray_max_valOut);


  void cal_SSM(
    HalconCpp::HTuple const& SSM_StdList,
    HalconCpp::HTuple const& SSM_threshold,
    HalconCpp::HTuple* single_exterior_point);


  void ContrastMean_4_8(
    HalconCpp::HObject const& GrayImage_cropped,
    HalconCpp::HTuple const& sub_height,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& ContrastMax_4,
    HalconCpp::HTuple const& ContrastMax_8,
    HalconCpp::HTuple* ContrastMatrix_4,
    HalconCpp::HTuple* ContrastMatrix_8,
    HalconCpp::HTuple* contrastMax_4Out,
    HalconCpp::HTuple* contrastMax_8Out);


  void copeland_count(
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& ScoreMatrix,
    HalconCpp::HTuple const& ExteriorPointRate,
    HalconCpp::HTuple* CopelandScoreList,
    HalconCpp::HTuple* MaxNumIndexSorted);


  void create_score_matrix(
    HalconCpp::HTuple const& ListLength,
    HalconCpp::HTuple const& StdList,
    HalconCpp::HTuple* ScoreMatrix);


  void cut_subimages_calculate_mm(
    HalconCpp::HObject const& Image_obj,
    HalconCpp::HTuple const& Height_obj,
    HalconCpp::HTuple const& sub_height,
    HalconCpp::HTuple const& Width_obj,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& ContrastMaxVal_4,
    HalconCpp::HTuple const& ContrastMaxVal_8,
    HalconCpp::HTuple const& GrayMaxVal,
    HalconCpp::HTuple const& iterations,
    HalconCpp::HTuple* ContrastMax_4Out,
    HalconCpp::HTuple* ContrastMax_8Out,
    HalconCpp::HTuple* GrayMaxOut,
    HalconCpp::HTuple* GrayMeanValueList,
    HalconCpp::HTuple* ContrastMeanValueList_4,
    HalconCpp::HTuple* ContrastMeanValueList_8);


  void cut_subimages_calculate_mm1(
    HalconCpp::HObject const& Image_obj,
    HalconCpp::HTuple const& Height_obj,
    HalconCpp::HTuple const& Width_obj,
    HalconCpp::HTuple const& sub_height,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& GrayMaxVal,
    HalconCpp::HTuple const& iterations,
    HalconCpp::HTuple* GrayMaxOut,
    HalconCpp::HTuple* GrayMeanValueList);


  void draw_contour_MMIV(
    HalconCpp::HObject* Contour,
    HalconCpp::HTuple const& MaxNumIndex,
    HalconCpp::HTuple const& col,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& sub_height);


  void draw_contour_MMIV1(
    HalconCpp::HObject* Contour,
    HalconCpp::HTuple const& GrayBordaMaxNumIndex,
    HalconCpp::HTuple const& index1,
    HalconCpp::HTuple const& col,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& sub_height);


  void maximin_count(
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& ScoreMatrix,
    HalconCpp::HTuple const& ExteriorPointRate,
    HalconCpp::HTuple* MaxScoreList,
    HalconCpp::HTuple* MaxNumIndexSorted);


  void power(
    HalconCpp::HTuple const& aa,
    HalconCpp::HTuple* bb);


  void zero_sloped_ransac(
    HalconCpp::HTuple const& MeanValList,
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& MaxVal,
    HalconCpp::HTuple const& iters,
    HalconCpp::HTuple const& epsilon,
    HalconCpp::HTuple const& threshold,
    HalconCpp::HTuple* StandardVarianceList,
    HalconCpp::HTuple* SSM_threshold);


    
};

#endif