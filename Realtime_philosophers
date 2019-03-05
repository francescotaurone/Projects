// DINING PHILOSOPHERS - Francesco Taurone 2017/18

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <string.h>

// CONSTANTS AND MACROS
// for readability
#define N_THREADS 10
#define N_SEATS 5
#define FOREVER for(;;)
#define TIMES 20
// number of phil life-cycles to be executed before return of the thread
#define NEIGH_INT 2
#define NEIGH_EXT 3

// DEFINITIONS OF NEW DATA TYPES
// for readability
typedef char thread_name_t[10];
typedef enum {THINKING, HUNGRY_STAND, HUNGRY_SIT, EATING, POLITE} state_t;

typedef struct {

    pthread_mutex_t m;
    pthread_cond_t phil_cv[N_THREADS];
    int where_sit[N_THREADS];
    int who_sit[N_SEATS];
    int who_was_polite[N_THREADS];
    int n_wait_chop;
    state_t state_phil[N_THREADS];
//for monitoring the eating situation while debugging
    long int  eating_situation[N_THREADS];
    int n_sit;

} monitor_t;

monitor_t mon;


//  MONITOR API
void pick_up(monitor_t *mon, int phil);
void put_down(monitor_t *mon, int phil);
//void seat_to_table(monitor_t *mon, int phil);
void monitor_init(monitor_t *mon);
void monitor_destroy(monitor_t *mon);
void *thread_type_1(void *arg);
double spend_some_time(int);
void think() { spend_some_time(100); }
void eat() { spend_some_time(100); }
int seat_left_int(int i, int times);
int seat_right_int(int i, int times);
int phil_left_ext(int i, int times);
int phil_right_ext(int i, int times);
int phil_left_int(monitor_t* mon, int phil, int times);
int phil_right_int(monitor_t* mon, int phil, int times);
void print_state(monitor_t *mon, int phil);
void sit_down(monitor_t *mon, int phil);
void stand_up(monitor_t *mon, int phil);


// IMPLEMENTATION OF MONITOR API
void sit_down(monitor_t *mon, int phil){
	int i, count_seat;
	pthread_mutex_lock (&mon->m);
	printf("Phil%d:	I am HUNGRY_STAND\n", phil);
	mon->state_phil[phil]=HUNGRY_STAND;
	print_state(mon, phil);
	while((mon->n_sit==N_SEATS)&&(mon->who_was_polite[phil]==N_THREADS+1)){
		pthread_cond_wait(&mon->phil_cv[phil], &mon->m);
	}

	//which means that somebody was polite with me
	if(mon->who_was_polite[phil]!=N_THREADS+1){
		mon->who_sit[mon->where_sit[mon->who_was_polite[phil]]]=phil;
		mon->where_sit[phil]=mon->where_sit[mon->who_was_polite[phil]];
		mon->where_sit[mon->who_was_polite[phil]]=N_SEATS+1;
		mon->state_phil[phil]=HUNGRY_SIT;
		mon->who_was_polite[phil]=N_THREADS+1;
		printf("POLITE BACK STAND Phil%d sitting in seat%d\n", phil, mon->where_sit[phil] );
	}
	//no one gave me his seats, therefore I'm looking for the free one that woke me up
	else{
		//I stop when I find it

		for(i=(phil*N_SEATS)/N_THREADS, count_seat=0;count_seat<N_SEATS&&(mon->where_sit[phil]==N_SEATS+1);i=seat_right_int(i, 1), count_seat++){
			if(mon->who_sit[i]==N_THREADS+1){
				mon->who_sit[i]=phil;
				mon->where_sit[phil]=i;
				mon->n_sit++;
				mon->state_phil[phil]=HUNGRY_SIT;
				printf("Phil%d sitting in seat%d\n", phil, i);
				}
		}
	}
	print_state(mon, phil);
	if(mon->state_phil[phil]==HUNGRY_STAND){
		printf("ERROR!\n");
	}
	pthread_mutex_unlock(&mon->m);
}
void stand_up(monitor_t *mon, int phil){
	int i, polite_check=0, check, was_polite=0;
	pthread_mutex_lock (&mon->m);
	i=phil;
	mon->state_phil[phil]=THINKING;
	printf("Phil%d: I am STANDING UP\n", phil);
//Check if I can give my seat to a preferred neighbour instead of freeing it
				while(polite_check<NEIGH_EXT){
					i=phil_left_ext(i, 1);
					polite_check++;
				}

				for(polite_check=0;polite_check<NEIGH_EXT*2-2; i=phil_right_ext(i, 1), polite_check++){
						if(i==phil_left_ext(phil, 1))
							i=phil_right_ext(phil, 2);
						if(mon->state_phil[i]==HUNGRY_STAND){
							//I found a waiting-to-seat neighbour, I sign my name in the table so that he can check where to go to sit
							printf("POLITE FORW STAND Phil%d signaling %d\n", phil, i );
							mon->who_was_polite[i]=phil;
							was_polite=1;
							pthread_cond_signal(&mon->phil_cv[i]);
							}
				}
						//Only if I didn't find any needing neighbour, I let the others try to get my seat
						if(!was_polite){

							mon->who_sit[mon->where_sit[phil]]=N_THREADS+1;
							mon->where_sit[phil]=N_SEATS+1;
							mon->n_sit--;
							//I start from here couse I have already checked my neighbours
							check=polite_check;
							while(check<N_THREADS){
								if(mon->state_phil[i]==HUNGRY_STAND){
									pthread_cond_signal(&mon->phil_cv[i]);
								}
								i=phil_right_ext(i, 1);
								check++;
							}

					}
			print_state(mon, phil);
			printf("Phil%d: I am THINKING\n", phil);
			pthread_mutex_unlock(&mon->m);

}
void print_state(monitor_t *mon, int phil){
	int i;
	printf("-----------\n");
	printf("Table state   :  ");
	for(i=0; i<N_SEATS; i++){
			if(mon->who_sit[i]==N_THREADS+1)
				printf(" ");
			else printf("%d", mon->state_phil[mon->who_sit[i]]);
		}
	printf("\n");
	printf("Table phil    :  ");
		for(i=0; i<N_SEATS; i++){
			if(mon->who_sit[i]==N_THREADS+1)
							printf(" ");
			else printf("%d", mon->who_sit[i]);

			}
		printf("\n");
		printf("---\n");
	printf("Phil seated   : ");
		for(i=0; i<N_THREADS; i++){
			if(mon->where_sit[i]==N_SEATS+1)
							printf(" ");
						else printf("%d", i);
			}
		printf("\n");
	printf("Phil walking  : ");
				for(i=0; i<N_THREADS; i++){
					if(mon->where_sit[i]==N_SEATS+1)
						printf("%d", i);
					else printf(" ");
					}
		printf("\n\n");
	printf("Phil state    : ");
						for(i=0; i<N_THREADS; i++){
							printf("%d", mon->state_phil[i]);
							}
				printf("\n");
		printf("-----------\n");

}

int seat_left_int(int i, int times){
	int count;
	int r;
	for(count=1; count<=times; count++){
		r=i-1;
		if(r<0)
			r = N_SEATS-1;
		else ;
	}
	return r;
}
int seat_right_int(int i, int times){
	int r;
	int count;
		for(count=1; count<=times; count++){
		r=i+1;
		if(r>=N_SEATS)
			r=  0;
		else ;
		}
			return r;
}
int phil_left_int(monitor_t *mon, int phil, int times){
	return mon->who_sit[seat_left_int(mon->where_sit[phil], times)];
}
int phil_right_int(monitor_t *mon, int phil, int times){
	return mon->who_sit[seat_right_int(mon->where_sit[phil], times)];
}
int phil_left_ext(int i, int times){
	int count;
		int r;
		for(count=1; count<=times; count++){
			r=i-1;
			if(r<0)
				r = N_THREADS-1;
			else ;
		}
		return r;
	}
int phil_right_ext(int i, int times){
	int count;
		int r;
		for(count=1; count<=times; count++){
			r=i-1;
			if(r<0)
				r = N_SEATS-1;
			else ;
		}
		return r;
}
void pick_up(monitor_t *mon, int phil) {
	printf("Phil%d:	I am HUNGRY_SIT\n", phil);

	pthread_mutex_lock(&mon->m);
//	mon->state_phil[mon->who_sit[phil]]=HUNGRY_SIT;
	print_state(mon, phil);
	mon->n_wait_chop++;

		//mon->n_wait_chop[seat]=mon->n_wait_chop[seat]+1;
	while(mon->state_phil[phil_left_int(mon, phil, 1)]==EATING||mon->state_phil[phil_right_int(mon, phil, 1)]==EATING) {
		if(mon->who_was_polite[phil]<N_THREADS+1){
			pthread_cond_signal(&mon->phil_cv[mon->who_was_polite[phil]]);
		}
		printf ("Phil%d:	I cannot pick up\n",phil);
		pthread_cond_wait(&mon->phil_cv[phil],&mon->m);
	}
	//qui è un po' ridondante fare doppiamente il controllo, ma in una situazione umana mi immagino che il filosofo polite
	//possa essee tale solo se rinuncia a mangiare per il waiting, non semplicemente quando ha fame

	//Probabilmente il polite dovrebbe essere tale solo con quelli a distanza 2, perché sono quelli che magari non stanno mangiando a causa sua


	mon->state_phil[phil]=EATING; //here change to phil
	mon->eating_situation[phil]++;
	mon->n_wait_chop--;

	printf("Phil%d:	I PICKED UP my chopsticks\n", phil);
	printf("Phil%d:	I am EATING\n", phil);
	print_state(mon, phil);
	pthread_mutex_unlock(&mon->m);

}

void put_down(monitor_t *mon, int phil) {
	int i, polite_check_before=0;

	pthread_mutex_lock(&mon->m);

	//mon->state_phil[phil]=THINKING;
	mon->state_phil[phil]=POLITE;



	//If I had the chopsticks cause someone was polite with me, I give back the control
	if(mon->who_was_polite[phil]<N_THREADS+1){
				pthread_cond_signal(&mon->phil_cv[mon->who_was_polite[phil]]);
				printf("POLITE BACK INT Phil%d signaling back Phil%d\n", phil, mon->who_was_polite[phil]);
			}
	else{


		if(mon->n_wait_chop>0){
			i=mon->where_sit[phil];
			for(polite_check_before=0;polite_check_before<NEIGH_INT; polite_check_before++){
				i=seat_left_int(i, 1);
			}
			//Check first your seat neighbourhood for letting them have priority
			for(polite_check_before=i;polite_check_before<NEIGH_INT*2-2; i=seat_right_int(i, 1), polite_check_before++){
					if(i==seat_left_int(mon->where_sit[phil], 1))
						i=seat_right_int(mon->where_sit[phil], 2);
					if(mon->state_phil[i]==HUNGRY_SIT){
						//mon->state_phil[seat]=POLITE;
						mon->who_was_polite[i]=phil;
						printf("POLITE FORW Phil%d signaling Phil%d\n", phil, i);
						pthread_cond_signal(&mon->phil_cv[i]);
						pthread_cond_wait(&mon->phil_cv[phil], &mon->m);
						mon->who_was_polite[i]=N_THREADS+1;
					}


				}
			}

	if(mon->state_phil[phil_left_int(mon, phil, 1)]==HUNGRY_SIT){
	pthread_cond_signal(&mon->phil_cv[phil_left_int(mon, phil, 1)]);
	printf("Phil%d seated in %d signaling %d\n",phil, mon->where_sit[phil], phil_left_int(mon, phil, 1));
	}
	if(mon->state_phil[phil_right_int(mon, phil, 1)]==HUNGRY_SIT){
	pthread_cond_signal(&mon->phil_cv[phil_right_int(mon, phil, 1)]);
	printf("Phil%d seated in %d signaling %d\n",phil, mon->where_sit[phil], phil_right_int(mon, phil, 1));

	}
	}
	pthread_mutex_unlock(&mon->m);

	printf("Phil%d:	I PUT DOWN my chopsticks \n", phil);

}

void monitor_init(monitor_t *mon) {
	// set initial value of monitor data structures, state variables, mutexes, counters, etc.
    // typically can use default attributes for monitor mutex and condvars
	int i;

	pthread_mutex_init(&mon->m,NULL);
	for (i=0; i<N_THREADS; i++){
			pthread_cond_init(&mon->phil_cv[i],NULL);

		    }
	for (i=0; i<N_THREADS; i++){
				mon->who_was_polite[i]=N_THREADS+1;
	    }
	for (i=0; i<N_THREADS; i++){
				mon->state_phil[i]=THINKING;
    }
    for (i=0; i<N_THREADS; i++){
    			mon->where_sit[i]=N_SEATS+1;
    	    }
    for (i=0; i<N_SEATS; i++){
    				mon->who_sit[i]=N_THREADS+1;
    	    }
    for (i=0; i<N_THREADS; i++){
        	mon->eating_situation[i]=0;
        }
    mon->n_wait_chop=0;
    mon->n_sit=0;
}

void monitor_destroy(monitor_t *mon) {
    // set initial value of monitor data structures, state variables, mutexes, counters, etc.
	int i;
	pthread_mutex_destroy(&mon->m);
	for (i=0; i<N_THREADS; i++){
			pthread_cond_destroy(&mon->phil_cv[i]);

		    }
	for (i=0; i<N_THREADS; i++){
				mon->who_was_polite[i]=N_THREADS+1;
	    }
	for (i=0; i<N_THREADS; i++){
				mon->state_phil[i]=THINKING;
    }
    for (i=0; i<N_THREADS; i++){
    			mon->where_sit[i]=N_SEATS+1;
    	    }
    for (i=0; i<N_SEATS; i++){
    				mon->who_sit[i]=N_THREADS+1;
    	    }
    for (i=0; i<N_THREADS; i++){
        	mon->eating_situation[i]=0;
        }
    mon->n_wait_chop=0;
    mon->n_sit=0;

}

// TYPE 1 THREAD LOOP
void *thread_type_1(void *arg) {
	// local variables definition and initialization
	int cycles, i;
	int thread_number;
	char *thread_name = (char *)arg; // thread_name = f(arg)
	thread_number=atoi(thread_name);
	printf("Phil%d: I have been created\n", thread_number);
	for(cycles=0; cycles<TIMES; cycles++){
	//FOREVER { // or any number of times

		think();
		sit_down(&mon, thread_number);
		pick_up(&mon, thread_number);
		eat();
		put_down(&mon, thread_number);
		stand_up(&mon, thread_number);

	}
	// SECTION: just for debugging purposes

	pthread_mutex_lock(&mon.m);
	printf("EATING SITUATION:   ");
	for(i=0; i<N_THREADS; i++){
		printf("%ld ", mon.eating_situation[i]);
	}
	printf("\n");
	pthread_mutex_unlock(&mon.m);
	// SECTION: just for debugging purposes
	pthread_exit(NULL);
}

// AUXILIARY FUNCTIONS
double spend_some_time(int max_steps) {
    double x, sum=0.0, step;
    long i, N_STEPS=rand()%(max_steps*1000000);
    step = 1/(double)N_STEPS;
    for(i=0; i<N_STEPS; i++) {
        x = (i+0.5)*step;
        sum+=4.0/(1.0+x*x);
    }
    return step*sum;
}

// MAIN FUNCTION
int main(void) {
    // thread management data structures
    pthread_t my_threads[N_THREADS];
    thread_name_t my_thread_names[N_THREADS];
    int i;

    // initialize monitor data structure before creating the threads
    monitor_init(&mon);

    for (i=0;i<N_THREADS;i++) {
     	sprintf(my_thread_names[i],"%d",i);
        // create N_THREADS thread with same entry point
        // these threads are distinguishable thanks to their argument (their name: "t1", "t2", ...)
        // thread names can also be used inside threads to show output messages
        pthread_create(&my_threads[i], NULL, thread_type_1, my_thread_names[i]);
    }

    for (i=0;i<N_THREADS;i++) {
        pthread_join(my_threads[i], NULL);
    }

    // free OS resources occupied by the monitor after creating the threads
    monitor_destroy(&mon);

    return EXIT_SUCCESS;
}
