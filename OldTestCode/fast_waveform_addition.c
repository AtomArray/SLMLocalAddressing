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

sem_t *mutex_locks[NUM_THREADS];


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
  for (int i = 0; i < NUM_SOURCE_WAVEFORMS; i++) {
	for (int j = 0 ; j < NUM_SAMPLES_IN_WAVEFORM; j++) {
	  source_waveforms[i][j] = offset * cos(2.0 * M_PI * freqs[i] * times[j]) + offset;
	}
  }
}

void sumWaveforms() {
  for (int i = 0; i < NUM_SOURCE_WAVEFORMS; i++) {
	for (int j = 0; j < NUM_SAMPLES_IN_WAVEFORM; j++) {
	  output_waveform[j] += source_waveforms[i][j];
	}
  }
}


void sumWaveforms2() {
  for (int i = 0; i < NUM_SOURCE_WAVEFORMS; i+= 4) {
	for (int j = 0; j < NUM_SAMPLES_IN_WAVEFORM; j++) {
	  output_waveform[j] += source_waveforms[i][j] + source_waveforms[i+1][j] + source_waveforms[i+2][j] + source_waveforms[i+3][j];
	}
  }
}


void addSourceWaveform(int index) {
  for (int j = 0; j < NUM_SAMPLES_IN_WAVEFORM; j++) {
	output_waveform[j] += source_waveforms[index][j];
  }
}

void sumWaveforms3() {
  for (int i = 0 ; i < NUM_SOURCE_WAVEFORMS; i++) {
	addSourceWaveform(i);
  }
}

void *sumWaveformsForThread(void *thread_index) {
  int thread = *((int *)thread_index);

  int initSource = thread * (NUM_SOURCE_WAVEFORMS / NUM_THREADS);
  int finalSource = (thread+1) * (NUM_SOURCE_WAVEFORMS / NUM_THREADS);



  for (int i = initSource; i < finalSource; i++) {
	for (int j = 0; j < NUM_SAMPLES_IN_WAVEFORM; j++) {
	  thread_waveforms[thread][j] += source_waveforms[i][j];
	}
  }

  return NULL;
}

void sumThreadWaveforms() {
  for (int i = 0; i < NUM_THREADS; i++) {
	for (int j = 0; j < NUM_SAMPLES_IN_WAVEFORM; j++) {
	  output_waveform[j] += thread_waveforms[i][j];
	}
  }
}

void sumWaveformsUsingThreads() {
  int use_threads = 1;
  

  pthread_t thread_ids[NUM_THREADS];
  int thread_indices[NUM_THREADS];

  for (int i = 0; i < NUM_THREADS; i++) {
	thread_indices[i] = i;

	if (use_threads) {
	  pthread_create(&thread_ids[i], NULL, sumWaveformsForThread, &thread_indices[i]);
	} else {
	  sumWaveformsForThread(&thread_indices[i]);
	}
  }

  if (use_threads) {
	for (int i = 0; i < NUM_THREADS; i++) {
	  pthread_join(thread_ids[i], NULL);
	}
  }
  sumThreadWaveforms();
}

float timeFunction(void (*fn)()) {
  clock_t t1 = clock();
  fn();
  clock_t t2 = clock();
  return (float)((t2 - t1) * 1000.0) / CLOCKS_PER_SEC;
}

int main() {
  float ms;
  for (int i = 0; i < NUM_THREADS; i++) {
	char str[10];
	memset(str, 0, 10);
	sprintf(str, "Thread%d", i);
	sem_unlink(str);
	mutex_locks[i] = sem_open(str, O_CREAT, 0644, 0);
  }



  ms = timeFunction(reset);
  printf("Time to reset: %f ms\n", ms);

  ms = timeFunction(generateSourceWaveforms);
  printf("Time to generate sources: %f ms\n", ms);

  ms = timeFunction(sumWaveforms2);
  //ms = timeFunction(sumWaveformsUsingThreads);
  printf("Time to sum waveforms: %f ms\n", ms);

  printf("First samples:\n");
  for (int i = 0; i < 20; i++) {
	printf("%u\n", output_waveform[i]);
  }

  return 0;
}
