// interface file for SWIG

%module OpticalFlow

// fbode SWIG fails on warnings, so make them non fatal
#pragma SWIG nowarn=321
#pragma SWIG nowarn=403
#pragma SWIG nowarn=325
#pragma SWIG nowarn=389
#pragma SWIG nowarn=341
#pragma SWIG nowarn=512
#pragma SWIG nowarn=362

typedef uint64_t size_t;


%{
#include <stdint.h>
#include <omp.h>

#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* #include <cflow/cpp/OpticalFlow.h> */
#include "OpticalFlow.h"

%}

/* %include "numpy.i" */
/* %init %{ */
/* import_array(); */
/* %} */
/* %apply (double* IN_ARRAY1, int DIM1) {(double* seq, int n)}; */

/* %include <cflow/cpp/OpticalFlow.h> */
%include "OpticalFlow.h"


/*******************************************************************
 * Python specific: numpy array <-> C++ pointer interface
 *******************************************************************/

#ifdef SWIGPYTHON

%{
PyObject *swig_ptr (PyObject *a)
{

    if (PyBytes_Check(a)) {
        return SWIG_NewPointerObj(PyBytes_AsString(a), SWIGTYPE_p_char, 0);
    }
    if (PyByteArray_Check(a)) {
        return SWIG_NewPointerObj(PyByteArray_AsString(a), SWIGTYPE_p_char, 0);
    }
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}

%}


%init %{
    /* needed, else crash at runtime */
    import_array();

%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(ctype, numpytype)

%{
PyObject * rev_swig_ptr(ctype *src, npy_intp size) {
    return PyArray_SimpleNewFromData(1, &size, numpytype, src);
}
%}

PyObject * rev_swig_ptr(ctype *src, size_t size);

%enddef

REV_SWIG_PTR(float, NPY_FLOAT32);
REV_SWIG_PTR(double, NPY_FLOAT64);
REV_SWIG_PTR(unsigned char, NPY_UINT8);
REV_SWIG_PTR(char, NPY_INT8);
REV_SWIG_PTR(unsigned short, NPY_UINT16);
REV_SWIG_PTR(short, NPY_INT16);
REV_SWIG_PTR(int, NPY_INT32);
REV_SWIG_PTR(unsigned int, NPY_UINT32);
REV_SWIG_PTR(int64_t, NPY_INT64);
REV_SWIG_PTR(uint64_t, NPY_UINT64);

#endif


