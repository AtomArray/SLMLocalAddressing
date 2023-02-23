#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>
#include <semaphore.h>

typedef unsigned short uint16_t;

static const int NUM_THREADS = 4;
static const int NUM_SOURCE_WAVEFORMS = 100;
static const int NUM_SAMPLES_IN_WAVEFORM = 100000;

uint16_t source_waveforms[NUM_SOURCE_WAVEFORMS][NUM_SAMPLES_IN_WAVEFORM];
uint16_t thread_waveforms[NUM_THREADS][NUM_SAMPLES_IN_WAVEFORM];

uint16_t output_waveform[NUM_SAMPLES_IN_WAVEFORM];


void reset() {
  memset(source_waveforms, 0, sizeof(uint16_t) * NUM_SOURCE_WAVEFORMS * NUM_SAMPLES_IN_WAVEFORM);
  memset(thread_waveforms, 0, sizeof(uint16_t) * NUM_THREADS * NUM_SAMPLES_IN_WAVEFORM);
  memset(output_waveform, 0, sizeof(uint16_t) * NUM_SAMPLES_IN_WAVEFORM);
}

void generateSourceWaveforms() {
  float freqs[NUM_SOURCE_WAVEFORMS];
  for (int i = 0; i < NUM_SOURCE_WAVEFORMS; i++) {
	freqs[i] = 1.0 + 0.01 * i;
  }

  float times[NUM_SAMPLES_IN_WAVEFORM];
  for (int i = 0; i < NUM_SAMPLES_IN_WAVEFORM; i++) {
	times[i] = 0.01 * i;
  }

  uint16_t offset = 1000;
  float f;
  for (int i = 0; i < NUM_SOURCE_WAVEFORMS; i++) {
	f = freqs[i];
	
	for (int j = 0 ; j < NUM_SAMPLES_IN_WAVEFORM; j++) {
	  source_waveforms[i][j] = offset * cos(2.0 * M_PI * f * times[j]) + offset;
	}
  }
}

float timeFunction(void (*fn)()) {
  clock_t t1 = clock();
  fn();
  clock_t t2 = clock();
  return (float)((t2 - t1) * 1000.0) / CLOCKS_PER_SEC;
}

int main() {
  float ms;

  ms = timeFunction(reset);
  printf("Time to reset: %f ms\n", ms);

  ms = timeFunction(generateSourceWaveforms);
  printf("Time to generate sources: %f ms\n", ms);

  return 0;
}
