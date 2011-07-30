attribute float aVis;
attribute vec3 aPos;
//attribute vec3 aNorm;
varying float vVis;
//varying vec3 vNorm;

void main(void) {
    //vNorm = aNorm;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(aPos, 1.0);
    vVis = aVis;
}
