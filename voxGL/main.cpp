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

#include "opengl.h"
#include "render.h"
#include "ResourceManager.h"

#include <SDL/SDL.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <algorithm>

#undef main

inline unsigned char randColor() {
	return rand()%255+1;
}

void buildWireframe() {
	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y)
			for(int z=0; z<DIM; ++z) {
				if( ( ((x&7)==0) && ((y&7)==0) ) || ( ((x&7)==0) && ((z&7)==0) ) || ( ((z&7)==0) && ((y&7)==0) )  )
					voxels[x][y][z] = randColor();
				else
					voxels[x][y][z] = 0;
			}
}

void buildParticles2() {
	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y)
			for(int z=0; z<DIM; ++z) {
				if( (x&3)==0 && (y&3)==0 && (z&3)==0)
					voxels[x][y][z] = randColor();
				else
					voxels[x][y][z] = 0;
			}
}

void buildParticles() {
	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y)
			for(int z=0; z<DIM; ++z) {
				if( (x&1) & (y&1) & (z&1))
					voxels[x][y][z] = randColor();
				else
					voxels[x][y][z] = 0;
			}
}


void buildBox() {
	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y) 
			for(int z=0; z<DIM; ++z) 
				voxels[x][y][z] = 0;

	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y) {
			voxels[x][y][DIM-1] = randColor();
			voxels[x][y][0] = randColor();
		}

		for(int x=0; x<DIM; ++x)
			for(int z=0; z<DIM; ++z) {
				voxels[x][0][z] = randColor();
				voxels[x][DIM-1][z] = randColor();
			}

			for(int y=0; y<DIM; ++y)
				for(int z=0; z<DIM; ++z) {
					voxels[DIM-1][y][z] = randColor();
				}
}

void buildRandom() {
	for(int x=0; x<DIM; ++x)
		for(int y=0; y<DIM; ++y) 
			for(int z=0; z<DIM; ++z) 
				voxels[x][y][z] = 0;

	for(int i=0; i<DIM*DIM*DIM>>4; ++i)
		voxels[rand()%DIM][rand()%DIM][rand()%DIM] = randColor();
}


int main(int argc, char* argv[]) {

	ResourceManager::init("assets.zip");

	/* First, initialize SDL's video subsystem. */
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 ) {
        fprintf( stderr, "Video initialization failed: %s\n", SDL_GetError( ) );
		exit(1);
    }

	printf("%dx%dx%D OpenGL Voxel demo\n\nARROWS and CTRL+ARROWS to moving\nESC to quit\n1-5 to select scene\n\n", DIM, DIM, DIM);

	char buf[255];
	sprintf(buf, "%dx%dx%d OpenGL Voxel demo", DIM, DIM, DIM);
	SDL_WM_SetCaption(buf, buf);

	/* Let's get some video information. */
    const SDL_VideoInfo* info = SDL_GetVideoInfo( );

    if( !info ) {
        fprintf( stderr, "Video query failed: %s\n", SDL_GetError( ) );
		exit(1);
    }

    SDL_GL_SetAttribute( SDL_GL_DEPTH_SIZE, 16 );
    SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );

#if USE_FULL_SCREEN
	int width = 0, height = 0;
    int flags = SDL_OPENGL | SDL_FULLSCREEN;
#else
	int width = 1024;
	int height = 576;
	int flags = SDL_OPENGL;
#endif

	SDL_Surface* mainSurface =  SDL_SetVideoMode( width, height, info->vfmt->BitsPerPixel, flags );
	// Set the video mode
    if( !mainSurface ) {
        fprintf( stderr, "Video mode set failed: %s\n", SDL_GetError( ) );
		exit(1);
    }
	width = mainSurface->w;
	height = mainSurface->h;

	if (glewInit() != GLEW_OK) {
		fprintf( stderr, "Error in glewInit\n" );
		exit(1);
	}

	static unsigned ticks =  SDL_GetTicks();

	Render::instance().reshape(width, height);
	buildWireframe();
	Render::instance().buildVBO();

	bool done = false;

	while(!done) {

		unsigned int t  = SDL_GetTicks();
		unsigned int dt = std::min(t-ticks, 250u);
		ticks=t;

		SDL_Event event;
		while( SDL_PollEvent( &event ) ) {
			switch( event.type ) {
				case SDL_MOUSEBUTTONDOWN:
					switch(event.button.button) {
						case 1:
						case 2:
						case 3:
						case 4:
						case 5:
							break;
					}
					break;
				case SDL_MOUSEBUTTONUP:
					switch(event.button.button) {
						case 1:
							break;
						case 3:
							break;
					}
				case SDL_MOUSEMOTION:
					break;

				case SDL_KEYDOWN:
					switch( event.key.keysym.sym ) {
						case SDLK_ESCAPE:
							done = true;
							break;

						case SDLK_SPACE:
							break;

						case SDLK_1:
							buildWireframe();
							Render::instance().buildVBO();
							break;
						case SDLK_2:
							buildParticles();
							Render::instance().buildVBO();
							break;
						case SDLK_3:
							buildBox();
							Render::instance().buildVBO();
							break;
						case SDLK_4:
							buildRandom();
							Render::instance().buildVBO();
							break;
						case SDLK_5:
							buildParticles2();
							Render::instance().buildVBO();
							break;

						case SDLK_F1:
						case SDLK_F2:
						case SDLK_F3:
						case SDLK_F4:
						case SDLK_F5:
						case SDLK_F6:
						case SDLK_F7:
						case SDLK_F8:
						case SDLK_F9:
						case SDLK_F10:
						case SDLK_F11:
							break;
						default:
							break;
					}
					break;
				case SDL_QUIT:
					/* Handle quit requests (like Ctrl-c). */
					SDL_Quit();
					return 0;
			}

		}

		Uint8 *keystatus = SDL_GetKeyState(NULL);
		float dx = 0, dy = 0, dz = 0, angvel = 0;

		float accel = 1 + 5*keystatus[SDLK_LSHIFT];
		if (!keystatus[SDLK_LCTRL]) {
			if (keystatus[SDLK_LEFT])	angvel = -1.0f * accel;
			if (keystatus[SDLK_RIGHT])	angvel =  1.0f * accel;
			if (keystatus[SDLK_UP])		dz =  0.1f * accel;
			if (keystatus[SDLK_DOWN])	dz = -0.1f * accel;
		} else {
			if (keystatus[SDLK_LEFT])	dx =  0.1f * accel;
			if (keystatus[SDLK_RIGHT])	dx =  -0.1f * accel;
			if (keystatus[SDLK_UP])		dy =  -0.1f * accel;
			if (keystatus[SDLK_DOWN])	dy =  0.1f * accel;
		}

		Render::instance().move(dx, dy, dz);
		Render::instance().turn(0, angvel, 0);

		Render::instance().draw();
		SDL_GL_SwapBuffers();

		static float fps = 0;
		const float factor=0.5f;
		if(dt>0)
			fps = fps*factor + (1000.0f/dt)*(1.0f - factor);

		printf("FPS=%0.2f           \r", fps);

		SDL_Delay(1);	
	}

	return 0;
}

