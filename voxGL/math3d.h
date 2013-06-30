/*  
	Copyright (c) 2013, Alexey Saenko
	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/ 

#ifndef MATH3D_H
#define MATH3D_H

#include <math.h>

#define EPSILON 1e-6f

inline bool is_NAN(float x) {
	return (*(unsigned int*)&x & 0x7f800000) == 0x7f800000;
}

inline  float absf(float v) { 
	unsigned int result = *(unsigned int*) &v;
	result &= 0x7fffffff;
    return *(float*)&result;	
}

inline  float signf(float v) {		// return 1.0 if v>=0, return -1 if v<0  
	unsigned int result = *(unsigned int*) &v;
	result = (result & 0x80000000) | 0x3f800000;
    return *(float*)&result;	
}

inline float fast_sqrtf(float x) {	// 5% calculation error
	unsigned int result = *(unsigned int*)&x;
	result = ( ((result - 0x3f800000) >> 1) + 0x3f800000 ) & 0x7fffffff;
	return *(float*)&result;
}

inline unsigned clp2(unsigned x) {	// round up to 2^n 
	x--;
	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	return x + 1;
}

inline unsigned flp2(unsigned x) {	// round down to 2^n 
	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	return x - (x >> 1);
}

const	float PI=3.14159265359f;
const	float PI_2=3.14159265359f/2.0f;
const	float PI_4=3.14159265359f/4.0f;
inline  float deg2rad(float a)			{ return PI/180.0f*a; }
inline  float rad2deg(float a)			{ return 180.0f/PI*a; }

struct vec2;
struct vec3;
struct vec4;
struct mat3;
struct mat4;
struct quat;

// vec2                                                                      

struct vec2 {
	
	inline vec2()                                   {}
	inline vec2(const float x, const float y) : x(x), y(y)      {}
	inline vec2(const float *v): x(v[0]), y(v[1])   {}
	inline vec2(const float v): x(v), y(v)          {}
	inline vec2(const vec2 &v): x(v.x), y(v.y)      {}
	inline vec2(const vec3 &v);
	
	inline int operator==(const vec2 &v) { return (absf(x - v.x) < EPSILON && absf(y - v.y) < EPSILON); }
	inline int operator!=(const vec2 &v) { return !(*this == v); }
	
	inline const vec2 operator*(float f) const { return vec2(x * f,y * f); }
	inline const vec2 operator/(float f) const { return vec2(x / f,y / f); }
	inline const vec2 operator+(const vec2 &v) const { return vec2(x + v.x,y + v.y); }
	inline const vec2 operator-() const { return vec2(-x,-y); }
	inline const vec2 operator-(const vec2 &v) const { return vec2(x - v.x,y - v.y); }
	
	inline vec2 &operator*=(float f) { return *this = *this * f; }
	inline vec2 &operator/=(float f) { return *this = *this / f; }
	inline vec2 &operator+=(const vec2 &v) { return *this = *this + v; }
	inline vec2 &operator-=(const vec2 &v) { return *this = *this - v; }
	
	inline float operator*(const vec2 &v) const { return x * v.x + y * v.y; }
	
	inline operator float*() { return (float*)&x; }
	inline operator const float*() const { return (float*)&x; }
	
	inline float &operator[](int i) { return v[i]; }
	inline float operator[](int i) const { return v[i]; }
	
	inline float length() const			{ return sqrtf(x * x + y * y); }
	inline float fast_length() const    { return fast_sqrtf(x * x + y * y); }
	inline float length2() const		{ return x * x + y * y; }

	inline float normalize() {
		float inv,length = sqrtf(x * x + y * y);
		inv = 1.0f / length;
		x *= inv;
		y *= inv;
		return length;
	}

	inline float fast_normalize() {
		float length = fast_sqrtf(x * x + y * y);
		float inv = 1.0f / length;
		x *= inv;
		y *= inv;
		return length;
	}

	inline float safe_normalize() {
		float inv,length = sqrtf(x * x + y * y);
		if(length < EPSILON*10) return length;  
		inv = 1.0f / length;
		x *= inv;
		y *= inv;
		return length;
	}
	
	
	union {
		struct {
			float x,y;
		};
		float v[2];
	};
};

// vec3                                                                      

struct vec3 {
	
	inline vec3()                                           {}
	inline vec3(const float x, const float y, const float z) : x(x), y(y), z(z) {}
	inline vec3(const float *v): x(v[0]), y(v[1]), z(v[2])  {}
	inline vec3(const float v) : x(v), y(v), z(v)           {}
	inline vec3(const vec2 &v) : x(v.x), y(v.y), z(0)		{}
	inline vec3(const vec2 &v, float az) : x(v.x), y(v.y), z(az)		{}
	inline vec3(const vec3 &v) : x(v.x), y(v.y), z(v.z)     {}
	inline vec3(const vec4 &v);
	
	inline int operator==(const vec3 &v) { return (absf(x - v.x) < EPSILON && absf(y - v.y) < EPSILON && absf(z - v.z) < EPSILON); }
	inline int operator!=(const vec3 &v) { return !(*this == v); }
	
	inline const vec3 operator*(float f) const { return vec3(x * f,y * f,z * f); }
	inline const vec3 operator/(float f) const { return vec3(x / f,y / f,z / f); }
	inline const vec3 operator+(const vec3 &v) const { return vec3(x + v.x,y + v.y,z + v.z); }
	inline const vec3 operator-() const { return vec3(-x,-y,-z); }
	inline const vec3 operator-(const vec3 &v) const { return vec3(x - v.x,y - v.y,z - v.z); }
	
	inline vec3 &operator*=(float f) { return *this = *this * f; }
	inline vec3 &operator/=(float f) { return *this = *this / f; }
	inline vec3 &operator+=(const vec3 &v) { return *this = *this + v; }
	inline vec3 &operator-=(const vec3 &v) { return *this = *this - v; }
	
	inline float operator*(const vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
	
	inline operator float*() { return (float*)&x; }
	inline operator const float*() const { return (float*)&x; }
	
	inline float &operator[](int i) { return v[i]; }
	inline float operator[](int i) const { return v[i]; }
	
	inline float length() const			{ return sqrtf(x * x + y * y + z * z); }
	inline float fast_length() const    { return fast_sqrtf(x * x + y * y + z * z); }
	inline float length2() const		{ return x * x + y * y + z * z; }

	inline float normalize() {
		float inv,length = sqrtf(x * x + y * y + z * z);
		inv = 1.0f / length;
		x *= inv;
		y *= inv;
		z *= inv;
		return length;
	}

	inline float fast_normalize() {
		float length = fast_sqrtf(x * x + y * y + z * z);
		float inv = 1.0f / length;
		x *= inv;
		y *= inv;
		z *= inv;
		return length;
	}

	inline float safe_normalize() {
		float inv,length = sqrtf(x * x + y * y + z * z);
		if(length < EPSILON*10) return length;  
		inv = 1.0f / length;
		x *= inv;
		y *= inv;
		z *= inv;
		return length;
	}
	
	inline void cross(const vec3 &v1,const vec3 &v2) {
		x = v1.y * v2.z - v1.z * v2.y;
		y = v1.z * v2.x - v1.x * v2.z;
		z = v1.x * v2.y - v1.y * v2.x;
	}
	
	union {
		struct {
			float x,y,z;
		};
		float v[3];
	};
};

inline vec2::vec2(const vec3 &v): x(v.x), y(v.y)      {}

inline vec3 cross(const vec3 &v1,const vec3 &v2) {
	vec3 ret;
	ret.x = v1.y * v2.z - v1.z * v2.y;
	ret.y = v1.z * v2.x - v1.x * v2.z;
	ret.z = v1.x * v2.y - v1.y * v2.x;
	return ret;
}

// vec4                                                                      

struct vec4 {
	
	inline vec4()                                                           {}
	inline vec4(const float x, const float y, const float z, const float w) : x(x), y(y), z(z), w(w)   {}
	inline vec4(const float *v) : x(v[0]), y(v[1]), z(v[2]), w(v[3])        {}
	inline vec4(const float v) : x(v), y(v), z(v), w(1.0f)                  {}
	inline vec4(const vec3 &v) : x(v.x), y(v.y), z(v.z), w(1)               {}
	inline vec4(const vec3 &v,float w) : x(v.x), y(v.y), z(v.z), w(w)       {}
	inline vec4(const vec4 &v) : x(v.x), y(v.y), z(v.z), w(v.w)             {}
	
	inline int operator==(const vec4 &v) { return (absf(x - v.x) < EPSILON && absf(y - v.y) < EPSILON && absf(z - v.z) < EPSILON && absf(w - v.w) < EPSILON); }
	inline int operator!=(const vec4 &v) { return !(*this == v); }
	
	inline const vec4 operator*(float f) const { return vec4(x * f,y * f,z * f,w * f); }
	inline const vec4 operator/(float f) const { return vec4(x / f,y / f,z / f,w / f); }
	inline const vec4 operator+(const vec4 &v) const { return vec4(x + v.x,y + v.y,z + v.z,w + v.w); }
	inline const vec4 operator-() const { return vec4(-x,-y,-z,-w); }
	inline const vec4 operator-(const vec4 &v) const { return vec4(x - v.x,y - v.y,z - v.z,z - v.w); }
	
	inline vec4 &operator*=(float f) { return *this = *this * f; }
	inline vec4 &operator/=(float f) { return *this = *this / f; }
	inline vec4 &operator+=(const vec4 &v) { return *this = *this + v; }
	inline vec4 &operator-=(const vec4 &v) { return *this = *this - v; }
	
	inline float operator*(const vec4 &v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }
	
	inline operator float*() { return (float*)&x; }
	inline operator const float*() const { return (float*)&x; }
	
	inline float &operator[](int i) { return v[i]; }
	inline float operator[](int i) const { return v[i]; }
	
	union {
		struct {
			float x,y,z,w;
		};
		float v[4];
	};
};

inline vec3::vec3(const vec4 &v) {
	x = v.x;
	y = v.y;
	z = v.z;
}

// mat2

struct mat2 {
	mat2() {}

	mat2(const float v) {
		mat[0] = v; mat[1] = 0;
		mat[2] = 0; mat[3] = v;
	}

	mat2(const float *m) {
		mat[0] = m[0]; mat[1] = m[1]; mat[2] = m[2]; mat[3] = m[3];
	}

	mat2(const mat2 &m) {
		mat[0] = m[0]; 	mat[1] = m[1];
		mat[2] = m[2]; 	mat[3] = m[3];
	}

	vec2 operator*(const vec2 &v) const {
		vec2 ret;
		ret[0] = mat[0] * v[0] + mat[2] * v[1];
		ret[1] = mat[1] * v[0] + mat[3] * v[1];
		return ret;
	}

	mat2 operator*(float f) const {
		mat2 ret;
		ret[0] = mat[0] * f; ret[1] = mat[1] * f;
		ret[2] = mat[2] * f; ret[3] = mat[3] * f;
		return ret;
	}

	mat2 operator*(const mat2 &m) const {
		mat2 ret;
		ret[0] = mat[0] * m[0] + mat[2] * m[1];
		ret[1] = mat[1] * m[0] + mat[3] * m[1];
		ret[2] = mat[0] * m[2] + mat[2] * m[3];
		ret[3] = mat[1] * m[2] + mat[3] * m[3];
		return ret;
	}

	mat2 operator+(const mat2 &m) const {
		mat2 ret;
		ret[0] = mat[0] + m[0];	ret[1] = mat[1] + m[1];
		ret[2] = mat[2] + m[2];	ret[3] = mat[3] + m[3];
		return ret;
	}

	mat2 operator-(const mat2 &m) const {
		mat2 ret;
		ret[0] = mat[0] - m[0]; ret[1] = mat[1] - m[1];
		ret[2] = mat[2] - m[2];	ret[3] = mat[3] - m[3];
		return ret;
	}

	mat2 &operator*=(float f) { return *this = *this * f; }
	mat2 &operator*=(const mat2 &m) { return *this = *this * m; }
	mat2 &operator+=(const mat2 &m) { return *this = *this + m; }
	mat2 &operator-=(const mat2 &m) { return *this = *this - m; }

	operator float*() { return mat; }
	operator const float*() const { return mat; }

	float &operator[](int i) { return mat[i]; }
	float operator[](int i) const { return mat[i]; }

	mat2 transpose() const {
		mat2 ret;
		ret[0] = mat[0];
		ret[1] = mat[2];
		ret[2] = mat[1];
		ret[3] = mat[3];
		return ret;
	}

	float det() const {
		return mat[0] * mat[3] - mat[1] * mat[2];
	}

	mat2 inverse() const {
		mat2 ret;
		float idet = 1.0f / det();
		ret[0] =  mat[3] * idet;
		ret[1] = -mat[1] * idet;
		ret[2] = -mat[2] * idet;
		ret[3] =  mat[0] * idet;
		return ret;
	}

	void zero() {
		mat[0] = mat[1] = mat[2] = mat[3] = 0.0f;
	}

	void identity() {
		mat[0] = 1; mat[1] = 0;
		mat[2] = 0; mat[3] = 1;
	}

	void rotate(float angle) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		mat[0] = c; mat[2] = -s;
		mat[1] = s; mat[3] = c;
	}

	void scale(const vec2 &v) {
		mat[0] = v.x; mat[1] = 0;
		mat[2] = 0;   mat[3] = v.y;
	}

	void scale(float x,float y) {
		scale(vec2(x,y));
	}

	static mat2 get_rotate(float angle) {
        mat2 ret;
        ret.rotate(angle);
        return ret;
	}

	static mat2 get_scale(const vec2 &v) {
        mat2 ret;
		ret.scale(v);
        return ret;
	}

	static mat2 get_scale(float x, float y) {
        mat2 ret;
		ret.scale(vec2(x,y));
        return ret;
	}

    static mat2 get_identity()  {
		return mat2(1.0f);
	}

    union {
	    float mat[4];
        float m[2][2];
    };
};

//  mat3                                                                      

struct mat3 {
	mat3() {}
	
	mat3(const float v) {
		mat[0] = v;   mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = v;   mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = v;
	}
	mat3(const float *m) {
		mat[0] = m[0]; mat[3] = m[3]; mat[6] = m[6];
		mat[1] = m[1]; mat[4] = m[4]; mat[7] = m[7];
		mat[2] = m[2]; mat[5] = m[5]; mat[8] = m[8];
	}
	mat3(const mat3 &m) {
		mat[0] = m[0]; mat[3] = m[3]; mat[6] = m[6];
		mat[1] = m[1]; mat[4] = m[4]; mat[7] = m[7];
		mat[2] = m[2]; mat[5] = m[5]; mat[8] = m[8];
	}

	mat3(const mat4 &m);

    mat3(const quat &q);
	
	vec2 operator*(const vec2 &v) const {
		vec2 ret;
		ret[0] = mat[0] * v[0] + mat[3] * v[1];
		ret[1] = mat[1] * v[0] + mat[4] * v[1];
		return ret;
	}

	vec3 operator*(const vec3 &v) const {
		vec3 ret;
		ret[0] = mat[0] * v[0] + mat[3] * v[1] + mat[6] * v[2];
		ret[1] = mat[1] * v[0] + mat[4] * v[1] + mat[7] * v[2];
		ret[2] = mat[2] * v[0] + mat[5] * v[1] + mat[8] * v[2];
		return ret;
	}

	vec4 operator*(const vec4 &v) const {
		vec4 ret;
		ret[0] = mat[0] * v[0] + mat[3] * v[1] + mat[6] * v[2];
		ret[1] = mat[1] * v[0] + mat[4] * v[1] + mat[7] * v[2];
		ret[2] = mat[2] * v[0] + mat[5] * v[1] + mat[8] * v[2];
		ret[3] = v[3];
		return ret;
	}

	mat3 operator*(float f) const {
		mat3 ret;
		ret[0] = mat[0] * f; ret[3] = mat[3] * f; ret[6] = mat[6] * f;
		ret[1] = mat[1] * f; ret[4] = mat[4] * f; ret[7] = mat[7] * f;
		ret[2] = mat[2] * f; ret[5] = mat[5] * f; ret[8] = mat[8] * f;
		return ret;
	}
	mat3 operator*(const mat3 &m) const {
		mat3 ret;
		ret[0] = mat[0] * m[0] + mat[3] * m[1] + mat[6] * m[2];
		ret[1] = mat[1] * m[0] + mat[4] * m[1] + mat[7] * m[2];
		ret[2] = mat[2] * m[0] + mat[5] * m[1] + mat[8] * m[2];
		ret[3] = mat[0] * m[3] + mat[3] * m[4] + mat[6] * m[5];
		ret[4] = mat[1] * m[3] + mat[4] * m[4] + mat[7] * m[5];
		ret[5] = mat[2] * m[3] + mat[5] * m[4] + mat[8] * m[5];
		ret[6] = mat[0] * m[6] + mat[3] * m[7] + mat[6] * m[8];
		ret[7] = mat[1] * m[6] + mat[4] * m[7] + mat[7] * m[8];
		ret[8] = mat[2] * m[6] + mat[5] * m[7] + mat[8] * m[8];
		return ret;
	}
	mat3 operator+(const mat3 &m) const {
		mat3 ret;
		ret[0] = mat[0] + m[0]; ret[3] = mat[3] + m[3]; ret[6] = mat[6] + m[6];
		ret[1] = mat[1] + m[1]; ret[4] = mat[4] + m[4]; ret[7] = mat[7] + m[7];
		ret[2] = mat[2] + m[2]; ret[5] = mat[5] + m[5]; ret[8] = mat[8] + m[8];
		return ret;
	}
	mat3 operator-(const mat3 &m) const {
		mat3 ret;
		ret[0] = mat[0] - m[0]; ret[3] = mat[3] - m[3]; ret[6] = mat[6] - m[6];
		ret[1] = mat[1] - m[1]; ret[4] = mat[4] - m[4]; ret[7] = mat[7] - m[7];
		ret[2] = mat[2] - m[2]; ret[5] = mat[5] - m[5]; ret[8] = mat[8] - m[8];
		return ret;
	}
	
	mat3 &operator*=(float f) { return *this = *this * f; }
	mat3 &operator*=(const mat3 &m) { return *this = *this * m; }
	mat3 &operator+=(const mat3 &m) { return *this = *this + m; }
	mat3 &operator-=(const mat3 &m) { return *this = *this - m; }
	
	operator float*() { return mat; }
	operator const float*() const { return mat; }
	
	float &operator[](int i) { return mat[i]; }
	float operator[](int i) const { return mat[i]; }
	
	mat3 transpose() const {
		mat3 ret;
		ret[0] = mat[0]; ret[3] = mat[1]; ret[6] = mat[2];
		ret[1] = mat[3]; ret[4] = mat[4]; ret[7] = mat[5];
		ret[2] = mat[6]; ret[5] = mat[7]; ret[8] = mat[8];
		return ret;
	}
	float det() const {
		float det;
		det = mat[0] * mat[4] * mat[8];
		det += mat[3] * mat[7] * mat[2];
		det += mat[6] * mat[1] * mat[5];
		det -= mat[6] * mat[4] * mat[2];
		det -= mat[3] * mat[1] * mat[8];
		det -= mat[0] * mat[7] * mat[5];
		return det;
	}
	mat3 inverse() const {
		mat3 ret;
		float idet = 1.0f / det();
		ret[0] =  (mat[4] * mat[8] - mat[7] * mat[5]) * idet;
		ret[1] = -(mat[1] * mat[8] - mat[7] * mat[2]) * idet;
		ret[2] =  (mat[1] * mat[5] - mat[4] * mat[2]) * idet;
		ret[3] = -(mat[3] * mat[8] - mat[6] * mat[5]) * idet;
		ret[4] =  (mat[0] * mat[8] - mat[6] * mat[2]) * idet;
		ret[5] = -(mat[0] * mat[5] - mat[3] * mat[2]) * idet;
		ret[6] =  (mat[3] * mat[7] - mat[6] * mat[4]) * idet;
		ret[7] = -(mat[0] * mat[7] - mat[6] * mat[1]) * idet;
		ret[8] =  (mat[0] * mat[4] - mat[3] * mat[1]) * idet;
		return ret;
	}
	
	void zero() {
		mat[0] = 0.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 0.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 0.0;
	}
	void identity() {
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	void rotate(float angle,const vec3 &axis) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		vec3 v = axis;
		v.normalize();
		float xy = v.x * v.y;
		float yz = v.y * v.z;
		float zx = v.z * v.x;
		float xs = v.x * s;
		float ys = v.y * s;
		float zs = v.z * s;
		mat[0] = (1.0f - c) * v.x * v.x + c; mat[3] = (1.0f - c) * xy - zs; 		mat[6] = (1.0f - c) * zx + ys;
		mat[1] = (1.0f - c) * xy + zs; 		 mat[4] = (1.0f - c) * v.y * v.y + c; 	mat[7] = (1.0f - c) * yz - xs;
		mat[2] = (1.0f - c) * zx - ys;		 mat[5] = (1.0f - c) * yz + xs; 		mat[8] = (1.0f - c) * v.z * v.z + c;
	}
	void rotate(float angle,float x,float y,float z) {
		rotate(angle,vec3(x,y,z));
	}
	void rotate_x(float angle) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = c; mat[7] = -s;
		mat[2] = 0.0; mat[5] = s; mat[8] = c;
	}
	void rotate_y(float angle) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		mat[0] = c; mat[3] = 0.0; mat[6] = s;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = -s; mat[5] = 0.0; mat[8] = c;
	}
	void rotate_z(float angle) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		mat[0] = c; mat[3] = -s; mat[6] = 0.0;
		mat[1] = s; mat[4] = c; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	void scale(const vec3 &v) {
		mat[0] = v.x; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = v.y; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = v.z;
	}
	void scale(float x,float y,float z) {
		scale(vec3(x,y,z));
	}
	void orthonormalize() {
		vec3 x(mat[0],mat[1],mat[2]);
		vec3 y(mat[3],mat[4],mat[5]);
		vec3 z;
		x.normalize();
		z.cross(x,y);
		z.normalize();
		y.cross(z,x);
		y.normalize();
		mat[0] = x.x; mat[3] = y.x; mat[6] = z.x;
		mat[1] = x.y; mat[4] = y.y; mat[7] = z.y;
		mat[2] = x.z; mat[5] = y.z; mat[8] = z.z;
	}


	static mat3 get_rotate(float angle,const vec3 &axis) {
        mat3 ret;
        ret.rotate(angle,axis);
        return ret;
    }

	static mat3 get_rotate(float angle,float x,float y,float z) {
        mat3 ret;
		ret.rotate(angle,vec3(x,y,z));
        return ret;
	}

	static mat3 get_rotate_x(float angle) {
        mat3 ret;
        ret.rotate_x(angle);
        return ret;
	}

	static mat3 get_rotate_y(float angle) {
        mat3 ret;
        ret.rotate_y(angle);
        return ret;
	}

	static mat3 get_rotate_z(float angle) {
        mat3 ret;
        ret.rotate_z(angle);
        return ret;
	}

	static mat3 get_scale(const vec3 &v) {
        mat3 ret;
		ret.scale(v);
        return ret;
	}

	static mat3 get_scale(float x,float y,float z) {
        mat3 ret;
		ret.scale(vec3(x,y,z));
        return ret;
	}

    static mat3 get_identity()  { return mat3(1.0f); }
	
    union {
	    float mat[9];
        float m[3][3];
//        struct {	
//          vec3 rot_x;
//			vec3 rot_y;
//			vec3 rot_z;
//		};
//        struct {	
//            vec3 rot[3];
//       };
    };
};

/*****************************************************************************/
/*                                                                           */
/* mat4                                                                      */
/*                                                                           */
/*****************************************************************************/

struct mat4 {
	
	mat4() {}

    mat4(const float v) {
		mat[0] = v;   mat[4] = 0.0; mat[8] = 0.0;  mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = v;   mat[9] = 0.0;  mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = v;   mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = v;
	}

	mat4(const mat3 &m) {
		mat[0] = m[0]; mat[4] = m[3]; mat[8] = m[6]; mat[12] = 0.0;
		mat[1] = m[1]; mat[5] = m[4]; mat[9] = m[7]; mat[13] = 0.0;
		mat[2] = m[2]; mat[6] = m[5]; mat[10] = m[8]; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	mat4(const float *m) {
		mat[0] = m[0]; mat[4] = m[4]; mat[8] = m[8]; mat[12] = m[12];
		mat[1] = m[1]; mat[5] = m[5]; mat[9] = m[9]; mat[13] = m[13];
		mat[2] = m[2]; mat[6] = m[6]; mat[10] = m[10]; mat[14] = m[14];
		mat[3] = m[3]; mat[7] = m[7]; mat[11] = m[11]; mat[15] = m[15];
	}
	mat4(const mat4 &m) {
		mat[0] = m[0]; mat[4] = m[4]; mat[8] = m[8]; mat[12] = m[12];
		mat[1] = m[1]; mat[5] = m[5]; mat[9] = m[9]; mat[13] = m[13];
		mat[2] = m[2]; mat[6] = m[6]; mat[10] = m[10]; mat[14] = m[14];
		mat[3] = m[3]; mat[7] = m[7]; mat[11] = m[11]; mat[15] = m[15];
	}
	
    mat4(float m0, float m1, float m2,  float m3,
         float m4, float m5, float m6,  float m7,
         float m8, float m9, float m10, float m11,
         float m12,float m13,float m14, float m15) {
        mat[0]=m0;   mat[1]=m1;   mat[2]=m2;   mat[3]=m3;
        mat[4]=m4;   mat[5]=m5;   mat[6]=m6;   mat[7]=m7;
        mat[8]=m8;   mat[9]=m9;   mat[10]=m10; mat[11]=m11;
        mat[12]=m12; mat[13]=m13; mat[14]=m14; mat[15]=m15;
    }   

    mat4(const vec3 &pos, const quat &q);

	vec2 operator*(const vec2 &v) const {
		vec2 ret;
		ret[0] = mat[0] * v[0] + mat[4] * v[1] + mat[12];
		ret[1] = mat[1] * v[0] + mat[5] * v[1] + mat[13];
		return ret;
	}

	vec3 operator*(const vec3 &v) const {
		vec3 ret;
		ret[0] = mat[0] * v[0] + mat[4] * v[1] + mat[8] * v[2] + mat[12];
		ret[1] = mat[1] * v[0] + mat[5] * v[1] + mat[9] * v[2] + mat[13];
		ret[2] = mat[2] * v[0] + mat[6] * v[1] + mat[10] * v[2] + mat[14];
		return ret;
	}
	vec4 operator*(const vec4 &v) const {
		vec4 ret;
		ret[0] = mat[0] * v[0] + mat[4] * v[1] + mat[8] * v[2] + mat[12] * v[3];
		ret[1] = mat[1] * v[0] + mat[5] * v[1] + mat[9] * v[2] + mat[13] * v[3];
		ret[2] = mat[2] * v[0] + mat[6] * v[1] + mat[10] * v[2] + mat[14] * v[3];
		ret[3] = mat[3] * v[0] + mat[7] * v[1] + mat[11] * v[2] + mat[15] * v[3];
		return ret;
	}
	mat4 operator*(float f) const {
		mat4 ret;
		ret[0] = mat[0] * f; ret[4] = mat[4] * f; ret[8] = mat[8] * f; ret[12] = mat[12] * f;
		ret[1] = mat[1] * f; ret[5] = mat[5] * f; ret[9] = mat[9] * f; ret[13] = mat[13] * f;
		ret[2] = mat[2] * f; ret[6] = mat[6] * f; ret[10] = mat[10] * f; ret[14] = mat[14] * f;
		ret[3] = mat[3] * f; ret[7] = mat[7] * f; ret[11] = mat[11] * f; ret[15] = mat[15] * f;
		return ret;
	}
	mat4 operator*(const mat4 &m) const {
		mat4 ret;
		ret[0] = mat[0] * m[0] + mat[4] * m[1] + mat[8] * m[2] + mat[12] * m[3];
		ret[1] = mat[1] * m[0] + mat[5] * m[1] + mat[9] * m[2] + mat[13] * m[3];
		ret[2] = mat[2] * m[0] + mat[6] * m[1] + mat[10] * m[2] + mat[14] * m[3];
		ret[3] = mat[3] * m[0] + mat[7] * m[1] + mat[11] * m[2] + mat[15] * m[3];
		ret[4] = mat[0] * m[4] + mat[4] * m[5] + mat[8] * m[6] + mat[12] * m[7];
		ret[5] = mat[1] * m[4] + mat[5] * m[5] + mat[9] * m[6] + mat[13] * m[7];
		ret[6] = mat[2] * m[4] + mat[6] * m[5] + mat[10] * m[6] + mat[14] * m[7];
		ret[7] = mat[3] * m[4] + mat[7] * m[5] + mat[11] * m[6] + mat[15] * m[7];
		ret[8] = mat[0] * m[8] + mat[4] * m[9] + mat[8] * m[10] + mat[12] * m[11];
		ret[9] = mat[1] * m[8] + mat[5] * m[9] + mat[9] * m[10] + mat[13] * m[11];
		ret[10] = mat[2] * m[8] + mat[6] * m[9] + mat[10] * m[10] + mat[14] * m[11];
		ret[11] = mat[3] * m[8] + mat[7] * m[9] + mat[11] * m[10] + mat[15] * m[11];
		ret[12] = mat[0] * m[12] + mat[4] * m[13] + mat[8] * m[14] + mat[12] * m[15];
		ret[13] = mat[1] * m[12] + mat[5] * m[13] + mat[9] * m[14] + mat[13] * m[15];
		ret[14] = mat[2] * m[12] + mat[6] * m[13] + mat[10] * m[14] + mat[14] * m[15];
		ret[15] = mat[3] * m[12] + mat[7] * m[13] + mat[11] * m[14] + mat[15] * m[15];
		return ret;
	}
	mat4 operator+(const mat4 &m) const {
		mat4 ret;
		ret[0] = mat[0] + m[0]; ret[4] = mat[4] + m[4]; ret[8] = mat[8] + m[8]; ret[12] = mat[12] + m[12];
		ret[1] = mat[1] + m[1]; ret[5] = mat[5] + m[5]; ret[9] = mat[9] + m[9]; ret[13] = mat[13] + m[13];
		ret[2] = mat[2] + m[2]; ret[6] = mat[6] + m[6]; ret[10] = mat[10] + m[10]; ret[14] = mat[14] + m[14];
		ret[3] = mat[3] + m[3]; ret[7] = mat[7] + m[7]; ret[11] = mat[11] + m[11]; ret[15] = mat[15] + m[15];
		return ret;
	}
	mat4 operator-(const mat4 &m) const {
		mat4 ret;
		ret[0] = mat[0] - m[0]; ret[4] = mat[4] - m[4]; ret[8] = mat[8] - m[8]; ret[12] = mat[12] - m[12];
		ret[1] = mat[1] - m[1]; ret[5] = mat[5] - m[5]; ret[9] = mat[9] - m[9]; ret[13] = mat[13] - m[13];
		ret[2] = mat[2] - m[2]; ret[6] = mat[6] - m[6]; ret[10] = mat[10] - m[10]; ret[14] = mat[14] - m[14];
		ret[3] = mat[3] - m[3]; ret[7] = mat[7] - m[7]; ret[11] = mat[11] - m[11]; ret[15] = mat[15] - m[15];
		return ret;
	}
	
	mat4 &operator*=(float f) { return *this = *this * f; }
	mat4 &operator*=(const mat4 &m) { return *this = *this * m; }
	mat4 &operator+=(const mat4 &m) { return *this = *this + m; }
	mat4 &operator-=(const mat4 &m) { return *this = *this - m; }
	
	operator float*() { return mat; }
	operator const float*() const { return mat; }
	
	float &operator[](int i) { return mat[i]; }
	float operator[](int i) const { return mat[i]; }
	
	mat4 rotation() const {
		mat4 ret;
		ret[0] = mat[0]; ret[4] = mat[4]; ret[8] = mat[8]; ret[12] = 0;
		ret[1] = mat[1]; ret[5] = mat[5]; ret[9] = mat[9]; ret[13] = 0;
		ret[2] = mat[2]; ret[6] = mat[6]; ret[10] = mat[10]; ret[14] = 0;
		ret[3] = 0; ret[7] = 0; ret[11] = 0; ret[15] = 1;
		return ret;
	}
	mat4 transpose() const {
		mat4 ret;
		ret[0] = mat[0]; ret[4] = mat[1]; ret[8] = mat[2]; ret[12] = mat[3];
		ret[1] = mat[4]; ret[5] = mat[5]; ret[9] = mat[6]; ret[13] = mat[7];
		ret[2] = mat[8]; ret[6] = mat[9]; ret[10] = mat[10]; ret[14] = mat[11];
		ret[3] = mat[12]; ret[7] = mat[13]; ret[11] = mat[14]; ret[15] = mat[15];
		return ret;
	}
	mat4 transpose_rotation() const {
		mat4 ret;
		ret[0] = mat[0]; ret[4] = mat[1]; ret[8] = mat[2]; ret[12] = mat[12];
		ret[1] = mat[4]; ret[5] = mat[5]; ret[9] = mat[6]; ret[13] = mat[13];
		ret[2] = mat[8]; ret[6] = mat[9]; ret[10] = mat[10]; ret[14] = mat[14];
		ret[3] = mat[3]; ret[7] = mat[7]; ret[14] = mat[14]; ret[15] = mat[15];
		return ret;
	}
	
	float det() const {
		float det;
		det = mat[0] * mat[5] * mat[10];
		det += mat[4] * mat[9] * mat[2];
		det += mat[8] * mat[1] * mat[6];
		det -= mat[8] * mat[5] * mat[2];
		det -= mat[4] * mat[1] * mat[10];
		det -= mat[0] * mat[9] * mat[6];
		return det;
	}
	
	mat4 inverse() const {
		mat4 ret;
		float idet = 1.0f / det();
		ret[0] =  (mat[5] * mat[10] - mat[9] * mat[6]) * idet;
		ret[1] = -(mat[1] * mat[10] - mat[9] * mat[2]) * idet;
		ret[2] =  (mat[1] * mat[6] - mat[5] * mat[2]) * idet;
		ret[3] = 0.0;
		ret[4] = -(mat[4] * mat[10] - mat[8] * mat[6]) * idet;
		ret[5] =  (mat[0] * mat[10] - mat[8] * mat[2]) * idet;
		ret[6] = -(mat[0] * mat[6] - mat[4] * mat[2]) * idet;
		ret[7] = 0.0;
		ret[8] =  (mat[4] * mat[9] - mat[8] * mat[5]) * idet;
		ret[9] = -(mat[0] * mat[9] - mat[8] * mat[1]) * idet;
		ret[10] =  (mat[0] * mat[5] - mat[4] * mat[1]) * idet;
		ret[11] = 0.0;
		ret[12] = -(mat[12] * ret[0] + mat[13] * ret[4] + mat[14] * ret[8]);
		ret[13] = -(mat[12] * ret[1] + mat[13] * ret[5] + mat[14] * ret[9]);
		ret[14] = -(mat[12] * ret[2] + mat[13] * ret[6] + mat[14] * ret[10]);
		ret[15] = 1.0;
		return ret;
	}
	
	void zero() {
		mat[0] = 0.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 0.0; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 0.0; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 0.0;
	}
	void identity() {
		mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 1.0; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate(float angle,const vec3 &axis) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		vec3 v = axis;
		v.normalize();
		float xy = v.x * v.y;
		float yz = v.y * v.z;
		float zx = v.z * v.x;
		float xs = v.x * s;
		float ys = v.y * s;
		float zs = v.z * s;
		mat[0] = (1.0f - c) * v.x * v.x + c; 	mat[4] = (1.0f - c) * xy - zs; 			mat[8] = (1.0f - c) * zx + ys; mat[12] = 0.0;
		mat[1] = (1.0f - c) * xy + zs; 			mat[5] = (1.0f - c) * v.y * v.y + c; 	mat[9] = (1.0f - c) * yz - xs; mat[13] = 0.0;
		mat[2] = (1.0f - c) * zx - ys; 			mat[6] = (1.0f - c) * yz + xs; 			mat[10] = (1.0f - c) * v.z * v.z + c; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate(float angle,float x,float y,float z) {
		rotate(angle,vec3(x,y,z));
	}
	void rotate_x(float angle) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = c; mat[9] = -s; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = s; mat[10] = c; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate_y(float angle) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		mat[0] = c; mat[4] = 0.0; mat[8] = s; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = -s; mat[6] = 0.0; mat[10] = c; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate_z(float angle) {
		float rad = deg2rad(angle);
		float c = cosf(rad);
		float s = sinf(rad);
		mat[0] = c; mat[4] = -s; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = s; mat[5] = c; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 1.0; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void scale(const vec3 &v) {
		mat[0] = v.x; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = v.y; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = v.z; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void scale(float x,float y,float z) {
		scale(vec3(x,y,z));
	}
	void translate(const vec3 &v) {
		mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = v.x;
		mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = v.y;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 1.0; mat[14] = v.z;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void translate(float x,float y,float z) {
		translate(vec3(x,y,z));
	}
	void reflect(const vec4 &plane) {
		float x = plane.x;
		float y = plane.y;
		float z = plane.z;
		float x2 = x * 2.0f;
		float y2 = y * 2.0f;
		float z2 = z * 2.0f;
		mat[0] = 1.0f - x * x2; mat[4] = -y * x2; mat[8] = -z * x2; mat[12] = -plane.w * x2;
		mat[1] = -x * y2; mat[5] = 1.0f - y * y2; mat[9] = -z * y2; mat[13] = -plane.w * y2;
		mat[2] = -x * z2; mat[6] = -y * z2; mat[10] = 1.0f - z * z2; mat[14] = -plane.w * z2;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void reflect(float x,float y,float z,float w) {
		reflect(vec4(x,y,z,w));
	}
	
	void perspective(float fov,float aspect,float znear,float zfar) {
		float y = tanf(fov * PI / 360.0f);
		float x = y * aspect;
		mat[0] = 1.0f / x; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 1.0f / y; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = -(zfar + znear) / (zfar - znear); mat[14] = -(2.0f * zfar * znear) / (zfar - znear);
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = -1.0; mat[15] = 0.0;
	}

	static mat4 get_perspective(float fov, float aspect, float znear, float zfar) {
		mat4 ret;
		ret.perspective(fov, aspect, znear, zfar);
		return ret;
	}

	void ortho(float left, float right, float bottom, float top, float near, float far)	{
	    float r_l = right - left;
	    float t_b = top - bottom;
	    float f_n = far - near;
	    mat[0] = 2.0f / r_l;
	    mat[12] = - (right + left) / r_l;
	    mat[5] = 2.0f / t_b;
	    mat[13] = - (top + bottom) / t_b;
	    mat[10] = - 2.0f / f_n;
	    mat[14] = - (far + near) / f_n;
	    mat[3]=mat[7]=mat[11]=0;
	    mat[1] = mat[2] = mat[4] = mat[6] = mat[8] = mat[9] = 0;
	  //  		mat[12] = mat[13] = mat[14] = 0.0f;
	    mat[15] = 1.0f;
	}

/*
	static mat4 look_at(const vec3 &eye,const vec3 &dir,const vec3 &up) {
		vec3 x,y,z;
		mat4 m;
		z = eye - dir;
		z.normalize();
		x.cross(up,z);
		x.normalize();
		y.cross(z,x);
		y.normalize();
		m[0] = x.x; m[4] = x.y; m[8]  = x.z;  m[12] = eye.x;
		m[1] = y.x; m[5] = y.y; m[9]  = y.z;  m[13] = eye.y;
		m[2] = z.x; m[6] = z.y; m[10] = z.z;  m[14] = eye.z;
		m[3] = 0.0; m[7] = 0.0; m[11] = 0.0; m[15] = 1.0;
		return m;
	}
*/
	static mat4 look_at(const vec3 &eye,const vec3 &dir,const vec3 &up) {
		vec3 x,y,z;
		mat4 m0,m1;
		z = eye - dir;
		z.normalize();
		x.cross(up,z);
		x.normalize();
		y.cross(z,x);
		y.normalize();
		m0[0] = x.x; m0[4] = x.y; m0[8] = x.z; m0[12] = 0.0;
		m0[1] = y.x; m0[5] = y.y; m0[9] = y.z; m0[13] = 0.0;
		m0[2] = z.x; m0[6] = z.y; m0[10] = z.z; m0[14] = 0.0;
		m0[3] = 0.0; m0[7] = 0.0; m0[11] = 0.0; m0[15] = 1.0;
		m1.translate(-eye);
		return m0 * m1;
	}

    static mat4  get_translate(const vec3 &v) {     
        mat4 ret;
        ret.translate(v);
        return ret;
	}

    static mat4  get_translate(float x,float y,float z) {
        mat4 ret;
        ret.translate(vec3(x,y,z));
        return ret;
	}

	static mat4 get_rotate(float angle,const vec3 &axis) {
        mat4 ret;
        ret.rotate(angle,axis);
        return ret;
    }

	static mat4 get_rotate(float angle,float x,float y,float z) {
        mat4 ret;
		ret.rotate(angle,vec3(x,y,z));
        return ret;
	}

	static mat4 get_rotate_x(float angle) {
        mat4 ret;
        ret.rotate_x(angle);
        return ret;
	}

	static mat4 get_rotate_y(float angle) {
        mat4 ret;
        ret.rotate_y(angle);
        return ret;
	}

	static mat4 get_rotate_z(float angle) {
        mat4 ret;
        ret.rotate_z(angle);
        return ret;
	}

	static mat4 get_scale(const vec3 &v) {
        mat4 ret;
		ret.scale(v);
        return ret;
	}

	static mat4 get_scale(float x,float y,float z) {
        mat4 ret;
		ret.scale(vec3(x,y,z));
        return ret;
	}

    static mat4 get_identity()  { return mat4(1.0f); }

    mat4 operator!() const;

    union {
	    float mat[16];
        float m[4][4];
//        struct {	
//            vec3 rot_x;
//			float	m03;
//			vec3 rot_y;
//			float	m13;
//			vec3 rot_z;
//			float	m23;
//			union {	
//              struct	{	vec3 pos;	 };
//				struct	{	float x,y,z; };
//			};
//			float	m33;
//		};
    };
};

inline mat3::mat3(const mat4 &m) {
	mat[0] = m[0]; mat[3] = m[4]; mat[6] = m[8];
	mat[1] = m[1]; mat[4] = m[5]; mat[7] = m[9];
	mat[2] = m[2]; mat[5] = m[6]; mat[8] = m[10];
}

/*****************************************************************************/
/*                                                                           */
/* quat                                                                      */
/*                                                                           */
/*****************************************************************************/

struct quat {
	
	quat()                                                                      { }
	quat(float v) : x(v), y(v), z(v), w(v)                                      { }
	quat(float ax, float ay, float az, float aw): x(ax), y(ay), z(az), w(aw)    { }

	quat(const vec3 &dir,float angle) {
		set(dir,angle);
	}
	
	quat(const mat4 &m) {
		float trace = m[0] + m[5] + m[10]; 	// I removed + 1.0f; see discussion with Ethan
		if( trace > 0 ) {					// I changed M_EPSILON to 0
			float s = 0.5f / sqrtf(trace+ 1.0f);
			w = 0.25f / s;
			x = ( m[9] - m[6] ) * s;
			y = ( m[2] - m[8] ) * s;
			z = ( m[4] - m[1] ) * s;
		} else {
			if ( m[0] > m[5] && m[0] > m[10] ) {
				float s = 2.0f * sqrtf( 1.0f + m[0] - m[5] - m[10]);
				w = (m[9] - m[6] ) / s;
				x = 0.25f * s;
				y = (m[1] + m[4] ) / s;
				z = (m[2] + m[8] ) / s;
			} else if (m[5] > m[10]) {
				float s = 2.0f * sqrtf( 1.0f + m[5] - m[0] - m[10]);
				w = (m[2] - m[8] ) / s;
				x = (m[1] + m[4] ) / s;
				y = 0.25f * s;
				z = (m[6] + m[9] ) / s;
			} else {
				float s = 2.0f * sqrtf( 1.0f + m[10] - m[0] - m[5] );
				w = (m[4] - m[1] ) / s;
				x = (m[2] + m[8] ) / s;
				y = (m[6] + m[9] ) / s;
				z = 0.25f * s;
			}
		}
	}

	operator float*() { return (float*)&x; }
	operator const float*() const { return (float*)&x; }
	
	float &operator[](int i) { return q[i]; }
	float operator[](int i) const { return q[i]; }
	
	quat operator*(const quat &q) const {
		quat ret;
		float num12 = (y * q.z) - (z * q.y);
		float num11 = (z * q.x) - (x * q.z);
		float num10 = (x * q.y) - (y * q.x);
		float num9 = ((x * q.x) + (y * q.y)) + (z * q.z);
		ret.x = ((x * q.w) + (q.x * w)) + num12;
		ret.y = ((y * q.w) + (q.y * w)) + num11;
		ret.z = ((z * q.w) + (q.z * w)) + num10;
		ret.w = (w * q.w) - num9;
		return ret;
	}

	vec3 operator*(const vec3 &v) const {
		return vec3(
			v.x*(1 - 2*y*y - 2*z*z) + v.y*(2*x*y - 2*w*z) + v.z*(2*x*z + 2*w*y),
			v.x*(2*x*y + 2*w*z) + v.y*(1 - 2*x*x - 2*z*z) + v.z*(2*y*z - 2*w*x),
			v.x*(2*x*z - 2*w*y) + v.y*(2*y*z + 2*w*x) + v.z*(1 - 2*x*x - 2*y*y) );
	}

	const quat operator-() const { return quat(-x,-y,-z,-w); }
	const quat operator*(float f) const { return quat(x * f,y * f,z * f,w * f); }
	quat &operator*=(float f) { return *this = *this * f; }
	quat &operator*=(const quat &q) { return *this = *this * q; }
	const quat operator+(const quat &q) const { return quat(x + q.x, y + q.y, z + q.z, w + q.w); }
	quat &operator+=(const quat &q) { return *this = *this + q; }
	
	inline float normalize() {
		float inv,length = sqrtf(x * x + y * y + z * z + w * w);
		inv = 1.0f / length;
		x *= inv;
		y *= inv;
		z *= inv;
		w *= inv;
		return length;
	}

	inline float fast_normalize() {
		float length = fast_sqrtf(x * x + y * y + z * z + w * w);
		float inv = 1.0f / length;
		x *= inv;
		y *= inv;
		z *= inv;
		w *= inv;
		return length;
	}

	void set(const vec3 &dir,float angle) {
		float length = dir.length();
		if(length != 0.0) {
            angle = deg2rad(angle) / 2.0f;
			float sinangle = sinf(angle) / length;
			x = dir.x * sinangle;
			y = dir.y * sinangle;
			z = dir.z * sinangle;
			w = cosf(angle);
		} else {
			x = y = z = 0.0;
			w = 1.0;
		}
	}

	void set(float x,float y,float z,float angle) {
		set(vec3(x,y,z),angle);
	}
	
	void rotate_x(float angle) {
		float a = deg2rad(angle) * 0.5f;
		y = z = 0;
		x = sinf(a);
		w = cosf(a);
	}

	void rotate_y(float angle) {
		float a = deg2rad(angle) * 0.5f;
		x = z = 0;
		y = sinf(a);
		w = cosf(a);
	}

	void rotate_z(float angle) {
		float a = deg2rad(angle) * 0.5f;
		x = y = 0;
		z = sinf(a);
		w = cosf(a);
	}

	static quat get_rotate(float angle,const vec3 &axis) {
        quat ret;
        ret.set(axis, angle);
        return ret;
    }

	static quat get_rotate(float angle,float x,float y,float z) {
        quat ret;
		ret.set(vec3(x,y,z), angle);
        return ret;
	}

	static quat get_rotate_x(float angle) {
        quat ret;
        ret.rotate_x(angle);
        return ret;
	}

	static quat get_rotate_y(float angle) {
        quat ret;
        ret.rotate_y(angle);
        return ret;
	}

	static quat get_rotate_z(float angle) {
        quat ret;
        ret.rotate_z(angle);
        return ret;
	}

	union {
		struct {
			float x,y,z,w;
		};
		float q[4];
	};
};


inline quat slerp(const quat &q0,const quat &q1,float t) {
	float k0, k1=1.0f, cosomega = q0.x * q1.x + q0.y * q1.y + q0.z * q1.z + q0.w * q1.w;
	if(cosomega < 0.0f) {
		cosomega = -cosomega;
        k1 = -1.0f;
	} 
	if(1.0 - cosomega > EPSILON) {
		float omega = acosf(cosomega);
		float sinomega = sinf(omega);
		k0 = sinf((1.0f - t) * omega) / sinomega;
		k1 *= sinf(t * omega) / sinomega;
	} else {
		k0 = 1.0f - t;
		k1 *= t;
	}
    return  q0 * k0 + q1 * k1;
}
	
inline float qot(const quat &q1, const quat &q2) {
	return q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
}

inline  vec3 calcNormal(const vec3 vert[3])                             { vec3 result=cross(vert[0]-vert[1],vert[1]-vert[2]); result.normalize(); return result; }
inline  vec3 calcNormal(const vec3 &v0, const vec3 &v1, const vec3 &v2) { vec3 result=cross(v0-v1,v1-v2); result.normalize(); return result; }

inline mat3::mat3(const quat &q) {   
    float wx, wy, wz, xx, yy, yz, xy, xz, zz, x2, y2, z2;
    x2 = q.x + q.x;
    y2 = q.y + q.y;
    z2 = q.z + q.z;
    xx = q.x * x2;   xy = q.x * y2;   xz = q.x * z2;
    yy = q.y * y2;   yz = q.y * z2;   zz = q.z * z2;
    wx = q.w * x2;   wy = q.w * y2;   wz = q.w * z2;

    mat[0]=1.0f-(yy+zz); mat[1]=xy-wz;        mat[2]=xz+wy;
    mat[3]=xy+wz;        mat[4]=1.0f-(xx+zz); mat[5]=yz-wx;
    mat[6]=xz-wy;        mat[7]=yz+wx;        mat[8]=1.0f-(xx+yy);
}

inline mat4::mat4(const vec3 &p, const quat &q) {   
    float wx, wy, wz, xx, yy, yz, xy, xz, zz, x2, y2, z2;
    x2 = q.x + q.x;
    y2 = q.y + q.y;
    z2 = q.z + q.z;
    xx = q.x * x2;   xy = q.x * y2;   xz = q.x * z2;
    yy = q.y * y2;   yz = q.y * z2;   zz = q.z * z2;
    wx = q.w * x2;   wy = q.w * y2;   wz = q.w * z2;

    mat[0]=1.0f-(yy+zz); mat[1]=xy-wz;        mat[2]=xz+wy;
    mat[4]=xy+wz;        mat[5]=1.0f-(xx+zz); mat[6]=yz-wx;
    mat[8]=xz-wy;        mat[9]=yz+wx;        mat[10]=1.0f-(xx+yy);

    mat[3] = mat[7] = mat[11] = 0;
    mat[15] = 1;
    mat[12] = p[0];
    mat[13] = p[1];
    mat[14] = p[2];
//	pos=p;
}

struct rect {
	float x, y, width, height;
	rect()																			{}
	rect(float ax, float ay, float w, float h): x(ax), y(ay), width(w), height(h)	{}
	bool 	inside(float px, float py)	{ 	return px>=x && px<=x+width && py>=y && py<=y+height; }

	bool 	intersect(const rect &r) 	{ 	return r.x < x+width && x < r.x+r.width && r.y < y+height && y < r.y+r.height;	}
	void	add(float ax, float ay)	{
		if(ax > x + width)
			width = ax - x;
		else if(ax < x) {
			width += x - ax;
			x = ax;
		}
		if(ay > y + height)
			height = ay - y;
		else if(ay < y) {
			height += y - ay;
			y = ay;
		}
	}
};

inline vec3 reflex(const vec3& vel, const vec3 &normal) {	// vel must be normalized!
	return  normal*(2.0f*(-vel*normal)) + vel;
}

inline vec2 reflex(const vec2& vel, const vec2 &normal) {	// vel must be normalized!
	return  normal*(2.0f*(-vel*normal)) + vel;
}

//inline float randf(float v0=0.0f, float vd=1.0f)   { return v0+(rand()*2.0f/RAND_MAX-1.0f)*vd;  }

float atan2(float y, float x); //Principal arc tangent of y/x, in the interval [-pi,+pi] radians.

inline bool linesIntersection(const vec2 &tp1,const vec2 &tp2,const vec2 &sc1,const vec2 &sc2, vec2 &result) {
	float z  = (tp2.y-tp1.y)*(sc1.x-sc2.x)-(sc1.y-sc2.y)*(tp2.x-tp1.x);
	if(absf(z) < EPSILON)
		return false;
	float a = ( (tp2.y-tp1.y)*(sc1.x-tp1.x)-(sc1.y-tp1.y)*(tp2.x-tp1.x) ) / z;
	float b = ( (sc1.y-tp1.y)*(sc1.x-sc2.x)-(sc1.y-sc2.y)*(sc1.x-tp1.x) ) / z;

	if( (0 <= a) && (a <= 1) && (0 <= b) && (b <= 1) ) {
		result = tp1 + (tp2 - tp1) * b;
		return true;
	}
	return false;
}

#endif 

