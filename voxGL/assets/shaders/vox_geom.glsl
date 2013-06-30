#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 24) out;


uniform mat4 transform;

in vec4 vcolor[];

//out vec3 normal;
out vec4 color;

void main () {

	const 	float shade[6] = float[](1.0, 0.8, 0.5, 0.6, 0.9, 0.7);

	const	vec3	vx = vec3(0.5, 0, 0);
	const	vec3	vy = vec3(0, 0.5, 0);
	const	vec3	vz = vec3(0, 0, 0.5);

	vec3	p = gl_in[0].gl_Position.xyz;
	
//	normal = vec3(0,0,-1);
	color = vcolor[0]*shade[0];

	gl_Position = transform * vec4( p - vz - vy - vx, 1 );
	EmitVertex();
	
	gl_Position = transform * vec4( p - vz - vy + vx, 1 );
	EmitVertex();
	
	gl_Position = transform * vec4( p - vz + vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p - vz + vy + vx, 1 );
	EmitVertex();

	EndPrimitive();		// face 1


//	normal = vec3(0,1,0);
	color = vcolor[0]*shade[1];

	gl_Position = transform * vec4( p - vz + vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p - vz + vy + vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz + vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz + vy + vx, 1 );
	EmitVertex();

	EndPrimitive();		// face 2


//	normal = vec3(1,0,0);
	color = vcolor[0]*shade[2];

	gl_Position = transform * vec4( p + vz + vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz + vy + vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz - vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz - vy + vx, 1 );
	EmitVertex();

	EndPrimitive();		// face 3


//	normal = vec3(0,-1,0);
	color = vcolor[0]*shade[3];

	gl_Position = transform * vec4( p + vz - vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz - vy + vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p - vz - vy - vx, 1 );
	EmitVertex();
	
	gl_Position = transform * vec4( p - vz - vy + vx, 1 );
	EmitVertex();

	EndPrimitive();		// face 4


//	normal = vec3(-1,0,0);
	color = vcolor[0]*shade[4];

	gl_Position = transform * vec4( p - vz - vy - vx, 1 );
	EmitVertex();
	
	gl_Position = transform * vec4( p - vz + vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz - vy - vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz + vy - vx, 1 );
	EmitVertex();

	EndPrimitive();		// face 5


//	normal = vec3(1,0,0);
	color = vcolor[0]*shade[5];

	gl_Position = transform * vec4( p - vz - vy + vx, 1 );
	EmitVertex();
	
	gl_Position = transform * vec4( p - vz + vy + vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz - vy + vx, 1 );
	EmitVertex();

	gl_Position = transform * vec4( p + vz + vy + vx, 1 );
	EmitVertex();

	EndPrimitive();		// face 6

	
}
