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

#include "Render.h"
#include "RenderResource.h"

#include "opengl.h"
#include "utils.h"

unsigned char voxels[DIM][DIM][DIM];

unsigned int voxelsCount = 0;

struct Vertex {
	vec3	pos, col;
	Vertex()											   {}
	Vertex(const vec3 &p, const vec3 &c): pos(p), col(c)   {}
};

static std::vector<RenderResource*>	resources;

RenderResource::RenderResource() {
	resources.push_back(this);
}

RenderResource::~RenderResource() {
	vector_fast_remove(resources, this);
	release();
}

Render::Render(): curShader(0) {}

Render::~Render() {
	release();
}

static Render *render = 0;
Render& Render::instance() {
	if(!render)
		render = new Render();
	return *render;
}

void Render::destroy() {
	if(render) {
		delete render;
		render = 0;
	}
}

void Render::release() {
	for(std::vector<RenderResource*>::iterator r = resources.begin(); r != resources.end(); ++r)
		(*r)->release();
}

void Render::reshape(int w, int h) {
	width = w;
	height = h;
	aspect = (float) width / (float) height;

	proj = mat4::get_perspective(60, aspect, 0.1f, 1000.0f);
	view = mat4::get_translate(-DIM/2, DIM/2, -DIM/2) * mat4::get_rotate_z(-90) * mat4::get_rotate_y(90);

	voxVertShader.load("vox_vert.glsl"); 
	voxGeomShader.load("vox_geom.glsl"); 
	voxFragShader.load("vox_frag.glsl");
	voxShaderProgram.attach(&voxVertShader, &voxFragShader, &voxGeomShader);
	voxShaderProgram.bindAttrib(ATTRIB_POSITION, "position");
	voxShaderProgram.bindAttrib(ATTRIB_COLOR, "color");
	voxShaderProgram.link();

}


void Render::setShader(ShaderProgram *sp) {
	if(curShader == sp)
		return;
	curShader = sp;
	if(curShader) 
		curShader->use();
	else
		ShaderProgram::unuse();
}

void Render::draw() {
	glViewport(0, 0, width, height);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	mat4 viewProj = proj * view;

	setShader(&voxShaderProgram);
	voxShaderProgram.uniform("transform", viewProj);

	vbo.bind();
	glEnableVertexAttribArray(ATTRIB_POSITION);
	glEnableVertexAttribArray(ATTRIB_COLOR);
	glVertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, false, sizeof(Vertex), ((Vertex*)0)->pos);
	glVertexAttribPointer(ATTRIB_COLOR, 3, GL_FLOAT, false, sizeof(Vertex), ((Vertex*)0)->col);
	glDrawArrays(GL_POINTS, 0, voxelsCount);
	glDisableVertexAttribArray(ATTRIB_POSITION);
	glDisableVertexAttribArray(ATTRIB_COLOR);
	vbo.unbind();

}

vec3 getColor(int c) {
	unsigned short r = ((c & 0x30) >> 1); 
	unsigned short g = ((c & 0xc) << 2);
	unsigned short b = ((c & 0x3) << 3);

	r |= (r>>2) | (r>>4);
	g |= (g>>2) | (g>>4);
	b |= (b>>2) | (b>>4);

	r |= ((c>>6) << 1);
	g |= ((c>>6) << 2);
	b |= ((c>>6) << 1);
	
	return vec3((float)r / 0x1f, (float)g / 0x3f, (float)b / 0x1f);
}


void Render::buildVBO() {
	voxelsCount = 0;
	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y) 
			for(int z=0; z<DIM; ++z)
				if(voxels[x][y][z]!=0)
					voxelsCount++;

	std::vector<Vertex>	verts(voxelsCount);

	int idx=0;
	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y) 
			for(int z=0; z<DIM; ++z) {
				unsigned char v = voxels[x][y][z];
				if(v==0)
					continue;
				verts[idx] = Vertex(vec3((float)x, (float)y, (float)z), getColor(v));
				idx++;
			}
	vbo.bind();
	vbo.setData(verts.size() * sizeof(Vertex), GL_STATIC_DRAW, &verts[0]);
	vbo.unbind();
}

void Render::move(float x, float y, float z) {
	view = mat4::get_translate(x, y, z) * view;
}

void Render::turn(float x, float y, float z) {
	if(fabsf(x) > EPSILON)
		view = mat4::get_rotate_x(x) * view;
	if(fabsf(y) > EPSILON)
		view = mat4::get_rotate_y(y) * view;
	if(fabsf(z) > EPSILON)
		view = mat4::get_rotate_z(z) * view;
}
