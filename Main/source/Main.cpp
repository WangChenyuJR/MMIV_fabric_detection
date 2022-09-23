/*****************************************************************************
 * File generated by HDevelop Version 20.11
 *
 * Do not modify!
 *****************************************************************************/

#include "Main.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <string>

#ifndef __APPLE__
  #include <hdevengine/HDevEngineCpp.h>
#else
  #include <HDevEngineCpp/HDevEngineCpp.h>
#endif

using namespace HalconCpp;
using namespace HDevEngineCpp;

namespace MMIV {

  std::string sgResourcePath;
  
  bool AddResourcePathToProcedurePath()
  {
    HDevEngineCpp::HDevEngine().AddProcedurePath(sgResourcePath.c_str());
    return true;
  }

  bool LazyInitProcedurePath()
  {
    static std::mutex lock;
    std::unique_lock<std::mutex> locker(lock);
    static const bool init = AddResourcePathToProcedurePath();
    return init;
  }

  void SetResourcePath(const char* resource_path)
  {
    sgResourcePath = resource_path;
    std::replace(sgResourcePath.begin(),sgResourcePath.end(), '\\','/');
    if(sgResourcePath.length() > 0 && sgResourcePath[sgResourcePath.length()-1]!='/')
    {
      sgResourcePath+="/";
    }
    AddResourcePathToProcedurePath();
  }

#ifdef _WIN32
  void SetResourcePath(const wchar_t* resource_path)
  {
    SetResourcePath(resource_path ? HString(resource_path).TextA() : NULL);
  }
#endif

  template <typename T>
  struct ParamHandler
  {
  };

  template <>
  struct ParamHandler<HalconCpp::HTuple>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HTuple const&                        parameter)
    {
      proc.SetInputCtrlParamTuple(name, parameter);
    }

    static HalconCpp::HTuple GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputCtrlParamTuple(name);
    }
  };

  template <>
  struct ParamHandler<HalconCpp::HObject>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HObject const&                       parameter)
    {
      proc.SetInputIconicParamObject(name, parameter);
    }

    static HalconCpp::HObject GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputIconicParamObject(name);
    }
  };

  template <>
  struct ParamHandler<HalconCpp::HTupleVector>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HTupleVector const&                  parameter)
    {
      proc.SetInputCtrlParamVector(name, parameter);
    }

    static HalconCpp::HTupleVector GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputCtrlParamVector(name);
    }
  };

  template <>
  struct ParamHandler<HalconCpp::HObjectVector>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HObjectVector const&                 parameter)
    {
      proc.SetInputIconicParamVector(name, parameter);
    }

    static HalconCpp::HObjectVector GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputIconicParamVector(name);
    }
  };


  HDevProgram GetProgram(std::string const& program_file)
  {
    static std::mutex lock;
    static std::map<std::string,HDevProgram> programs;

    std::unique_lock<std::mutex> locker(lock);

    auto prog_iter = programs.find(program_file);
    if(prog_iter != programs.end())
    {
      return prog_iter->second;
    }
    else
    {
      HDevProgram program(program_file.c_str());
      programs[program_file] = program;
      return program;
    }
    return HDevProgram();
  }

  void borda_count(
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& ScoreMatrix,
    HalconCpp::HTuple const& ExteriorPointRate,
    HalconCpp::HTuple* BordaScoreList,
    HalconCpp::HTuple* MaxNumIndexSorted)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"borda_count");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"subimg_num",subimg_num);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ScoreMatrix",ScoreMatrix);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ExteriorPointRate",ExteriorPointRate);
        
    call.Execute();

    if (BordaScoreList != nullptr)
      * BordaScoreList = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"BordaScoreList");
    if (MaxNumIndexSorted != nullptr)
      * MaxNumIndexSorted = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"MaxNumIndexSorted");
  }

  void cal_common_divisor(
    HalconCpp::HTuple const& input1,
    HalconCpp::HTuple const& input2,
    HalconCpp::HTuple* common_divisor)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"cal_common_divisor");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"input1",input1);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"input2",input2);
        
    call.Execute();

    if (common_divisor != nullptr)
      * common_divisor = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"common_divisor");
  }

  void cal_imgcrop_gray_mean(
    HalconCpp::HObject const& ImageObj_cropped,
    HalconCpp::HTuple const& gray_max_val,
    HalconCpp::HTuple* Gray_Mean,
    HalconCpp::HTuple* gray_max_valOut)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"cal_imgcrop_gray_mean");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HObject>::SetParameter(call,"ImageObj_cropped",ImageObj_cropped);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"gray_max_val",gray_max_val);
        
    call.Execute();

    if (Gray_Mean != nullptr)
      * Gray_Mean = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"Gray_Mean");
    if (gray_max_valOut != nullptr)
      * gray_max_valOut = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"gray_max_valOut");
  }

  void cal_SSM(
    HalconCpp::HTuple const& SSM_StdList,
    HalconCpp::HTuple const& SSM_threshold,
    HalconCpp::HTuple* single_exterior_point)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"cal_SSM");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"SSM_StdList",SSM_StdList);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"SSM_threshold",SSM_threshold);
        
    call.Execute();

    if (single_exterior_point != nullptr)
      * single_exterior_point = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"single_exterior_point");
  }

  void ContrastMean_4_8(
    HalconCpp::HObject const& GrayImage_cropped,
    HalconCpp::HTuple const& sub_height,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& ContrastMax_4,
    HalconCpp::HTuple const& ContrastMax_8,
    HalconCpp::HTuple* ContrastMatrix_4,
    HalconCpp::HTuple* ContrastMatrix_8,
    HalconCpp::HTuple* contrastMax_4Out,
    HalconCpp::HTuple* contrastMax_8Out)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"ContrastMean_4_8");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HObject>::SetParameter(call,"GrayImage_cropped",GrayImage_cropped);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_height",sub_height);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_width",sub_width);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ContrastMax_4",ContrastMax_4);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ContrastMax_8",ContrastMax_8);
        
    call.Execute();

    if (ContrastMatrix_4 != nullptr)
      * ContrastMatrix_4 = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"ContrastMatrix_4");
    if (ContrastMatrix_8 != nullptr)
      * ContrastMatrix_8 = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"ContrastMatrix_8");
    if (contrastMax_4Out != nullptr)
      * contrastMax_4Out = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"contrastMax_4Out");
    if (contrastMax_8Out != nullptr)
      * contrastMax_8Out = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"contrastMax_8Out");
  }

  void copeland_count(
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& ScoreMatrix,
    HalconCpp::HTuple const& ExteriorPointRate,
    HalconCpp::HTuple* CopelandScoreList,
    HalconCpp::HTuple* MaxNumIndexSorted)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"copeland_count");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"subimg_num",subimg_num);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ScoreMatrix",ScoreMatrix);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ExteriorPointRate",ExteriorPointRate);
        
    call.Execute();

    if (CopelandScoreList != nullptr)
      * CopelandScoreList = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"CopelandScoreList");
    if (MaxNumIndexSorted != nullptr)
      * MaxNumIndexSorted = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"MaxNumIndexSorted");
  }

  void create_score_matrix(
    HalconCpp::HTuple const& ListLength,
    HalconCpp::HTuple const& StdList,
    HalconCpp::HTuple* ScoreMatrix)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"create_score_matrix");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ListLength",ListLength);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"StdList",StdList);
        
    call.Execute();

    if (ScoreMatrix != nullptr)
      * ScoreMatrix = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"ScoreMatrix");
  }

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
    HalconCpp::HTuple* ContrastMeanValueList_8)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"cut_subimages_calculate_mm");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HObject>::SetParameter(call,"Image_obj",Image_obj);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"Height_obj",Height_obj);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_height",sub_height);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"Width_obj",Width_obj);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_width",sub_width);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ContrastMaxVal_4",ContrastMaxVal_4);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ContrastMaxVal_8",ContrastMaxVal_8);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"GrayMaxVal",GrayMaxVal);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"iterations",iterations);
        
    call.Execute();

    if (ContrastMax_4Out != nullptr)
      * ContrastMax_4Out = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"ContrastMax_4Out");
    if (ContrastMax_8Out != nullptr)
      * ContrastMax_8Out = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"ContrastMax_8Out");
    if (GrayMaxOut != nullptr)
      * GrayMaxOut = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"GrayMaxOut");
    if (GrayMeanValueList != nullptr)
      * GrayMeanValueList = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"GrayMeanValueList");
    if (ContrastMeanValueList_4 != nullptr)
      * ContrastMeanValueList_4 = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"ContrastMeanValueList_4");
    if (ContrastMeanValueList_8 != nullptr)
      * ContrastMeanValueList_8 = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"ContrastMeanValueList_8");
  }

  void cut_subimages_calculate_mm1(
    HalconCpp::HObject const& Image_obj,
    HalconCpp::HTuple const& Height_obj,
    HalconCpp::HTuple const& Width_obj,
    HalconCpp::HTuple const& sub_height,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& GrayMaxVal,
    HalconCpp::HTuple const& iterations,
    HalconCpp::HTuple* GrayMaxOut,
    HalconCpp::HTuple* GrayMeanValueList)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"cut_subimages_calculate_mm1");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HObject>::SetParameter(call,"Image_obj",Image_obj);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"Height_obj",Height_obj);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"Width_obj",Width_obj);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_height",sub_height);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_width",sub_width);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"GrayMaxVal",GrayMaxVal);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"iterations",iterations);
        
    call.Execute();

    if (GrayMaxOut != nullptr)
      * GrayMaxOut = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"GrayMaxOut");
    if (GrayMeanValueList != nullptr)
      * GrayMeanValueList = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"GrayMeanValueList");
  }

  void draw_contour_MMIV(
    HalconCpp::HObject* Contour,
    HalconCpp::HTuple const& MaxNumIndex,
    HalconCpp::HTuple const& col,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& sub_height)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"draw_contour_MMIV");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"MaxNumIndex",MaxNumIndex);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"col",col);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_width",sub_width);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_height",sub_height);
        
    call.Execute();

    if (Contour != nullptr)
      * Contour = ParamHandler<HalconCpp::HObject>::GetParameter(call,"Contour");
  }

  void draw_contour_MMIV1(
    HalconCpp::HObject* Contour,
    HalconCpp::HTuple const& GrayBordaMaxNumIndex,
    HalconCpp::HTuple const& index1,
    HalconCpp::HTuple const& col,
    HalconCpp::HTuple const& sub_width,
    HalconCpp::HTuple const& sub_height)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"draw_contour_MMIV1");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"GrayBordaMaxNumIndex",GrayBordaMaxNumIndex);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"index1",index1);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"col",col);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_width",sub_width);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"sub_height",sub_height);
        
    call.Execute();

    if (Contour != nullptr)
      * Contour = ParamHandler<HalconCpp::HObject>::GetParameter(call,"Contour");
  }

  void maximin_count(
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& ScoreMatrix,
    HalconCpp::HTuple const& ExteriorPointRate,
    HalconCpp::HTuple* MaxScoreList,
    HalconCpp::HTuple* MaxNumIndexSorted)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"maximin_count");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"subimg_num",subimg_num);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ScoreMatrix",ScoreMatrix);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"ExteriorPointRate",ExteriorPointRate);
        
    call.Execute();

    if (MaxScoreList != nullptr)
      * MaxScoreList = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"MaxScoreList");
    if (MaxNumIndexSorted != nullptr)
      * MaxNumIndexSorted = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"MaxNumIndexSorted");
  }

  void power(
    HalconCpp::HTuple const& aa,
    HalconCpp::HTuple* bb)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"power");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"aa",aa);
        
    call.Execute();

    if (bb != nullptr)
      * bb = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"bb");
  }

  void zero_sloped_ransac(
    HalconCpp::HTuple const& MeanValList,
    HalconCpp::HTuple const& subimg_num,
    HalconCpp::HTuple const& MaxVal,
    HalconCpp::HTuple const& iters,
    HalconCpp::HTuple const& epsilon,
    HalconCpp::HTuple const& threshold,
    HalconCpp::HTuple* StandardVarianceList,
    HalconCpp::HTuple* SSM_threshold)
  {     
    static HDevEngineCpp::HDevProcedure procedure(GetProgram(sgResourcePath+"Main.hdev"),"zero_sloped_ransac");
    HDevEngineCpp::HDevProcedureCall call=procedure;
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"MeanValList",MeanValList);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"subimg_num",subimg_num);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"MaxVal",MaxVal);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"iters",iters);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"epsilon",epsilon);
    ParamHandler<HalconCpp::HTuple>::SetParameter(call,"threshold",threshold);
        
    call.Execute();

    if (StandardVarianceList != nullptr)
      * StandardVarianceList = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"StandardVarianceList");
    if (SSM_threshold != nullptr)
      * SSM_threshold = ParamHandler<HalconCpp::HTuple>::GetParameter(call,"SSM_threshold");
  }

};
