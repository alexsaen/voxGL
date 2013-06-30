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

#ifndef RENDER_H
#define RENDER_H

#include "math3d.h"
#include "VBO.h"
#include "Shader.h"
#include <vector>

class RenderResource;

static const unsigned int ATTRIB_POSITION = 0;
static const unsigned int ATTRIB_COLOR	  = 2;

const int DIM = 128;
extern unsigned char voxels[DIM][DIM][DIM];

class Render {
	friend class RenderResource;
	int		width, height;
	float	aspect, scale;

	mat4			view, proj;
	ShaderProgram	*curShader;


	VertexShader	voxVertShader;
	FragmentShader	voxFragShader;
	GeometryShader	voxGeomShader;
	ShaderProgram	voxShaderProgram;

	VBOVertex		vbo;

					Render();
	virtual			~Render();
public:

	static	Render	&instance();
	static	void	destroy();

			void	reshape(int w, int h);
	static	void	release();

	void	draw();

	void	setShader(ShaderProgram *sp);

	void	buildVBO();

	void	move(float x, float y, float z);
	void	turn(float x, float y, float z);

};

#endif 
