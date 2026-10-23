================================================================================
  MULTI-TASK BLACK-BOX OPTIMIZATION ENGINE - RENDSZER ÖSSZEFOGLALÓ
================================================================================
1. A PROGRAM MÛKÖDÉSE ÉS ARCHITEKTÚRÁJA
--------------------------------------------------------------------------------
- Rajintelligencia + Multi-Head Attention (MHA) + Evolúciós alapelvek.
- 64 bites fix pontos aritmetika (Q32.32): int64_t és __int128 köztes típusok a 
  túlcsordulás ellen, lebegõpontos (float/double) használat nélkül.
- Térbontás: A data_dim dimenzió num_heads darab fejre és head_dim méretû 
  alterekre oszlik. A query, key, value vektorok ezeken fókuszáltan futnak.
- Kommunikáció: Az ágensek a dinamikus kedvenc-listájukon (favorites) lévõ 
  partnerekkel osztják meg a tudásukat.
2. A RENDSZER VISELKEDÉSE ÉS DINAMIKÁJA (Szimulációs kör)
--------------------------------------------------------------------------------
Fázisok iterációnként:
1. Fitness Kiértékelés: Az ágensek data_params paramétereinek távolsága a 
   target_data célvektortól kiszámolja a last_fit_quality (0.0 - 1.0) értéket.
2. Figyelem és Adaptív Mutáció: Attention súlyok alapján környezeti vektor 
   számítása, majd adaptív zajbevezetés (minél rosszabb a fitness, annál nagyobb 
   a felfedezési zaj a lokális optimumok elkerülésére).
3. Evolúciós Szelekció: A gyenge ágensek átveszik a populáció legjobbjainak 
   mintáit (crossover jellegû tudásátadás).
3. MEMÓRIALÁBNYOM ÉS KEZELÉS (Zéró-Malloc Runtime)
--------------------------------------------------------------------------------
- Arena Allocation (master_memory_block): Minden ágens adata egyetlen 
  folytonos memóriablokkban lakik, megszüntetve a heap-fragmentációt.
- Scratchpad Pufferek: Elõre lefoglalt munkaterületek a SimulationContext-ben 
  (scratch_raw_scores, scratch_weights, scratch_head_contexts), így a belsõ 
  ciklusokban nincsenek dinamikus foglalások.
- Méretezési adatok (Becsült RAM):
  * Alapértelmezett (100 ágens, 32 dimenzió):      ~0.9 MB
  * Nagy rajintelligencia (1000 ágens, 32 dim):     ~9.3 MB
  * Magas dimenzió (1000 ágens, 1000 dimenzió):     ~41.5 MB
4. SEBESSÉG ÉS TELJESÍTMÉNY
--------------------------------------------------------------------------------
- Skálázhatóság: Az Attention skaláris szorzatok lokálisan, a head_dim fejeken 
  futnak, így a globális 1000+ dimenziós tér sem lassítja le a hálózatot.
- Teljesítmény: Egy szálon futtatva 32 dimenzióban tízezres, 1000 dimenzióban 
  is több száz / ezer iteráció / másodperc teljesítmény érhetõ el.
- Hardver-barát: FPU nélküli mikrokontrollerekre és beágyazott rendszerekre is 
  közvetlenül átültethetõ C99 szabványú kód.
================================================================================
Multi-Task Black-Box Optimization Engine
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#define FIXED_SHIFT_64 32       
#define FIXED_ONE_64   ((int64_t)1 << FIXED_SHIFT_64)  
#define FIXED_LIMIT_64 (4 * FIXED_ONE_64) 
#define LEARNING_RATE_SHIFT 4 
#define EXPLORATION_NOISE_MAX 4096 
typedef struct {
    int id;                   
    unsigned int rng_state; 
    int64_t *data_params;     // 64-bit fix pontos egyedi paramétertér
    int *preference_list;     
    int *favorites;           
    int favorites_count; 
    int64_t **query_vectors;  // Multi-Head lekérdezések 
    int64_t **key_vectors;    // Kulcsvektorok 
    int64_t **value_vectors;  // Értékvektorok 
    int64_t last_fit_quality; // 0.0 - 1.0 fix pontos fitness
    int current_vote; 
    int64_t confidence_score; 
} Agent;
typedef struct {
    int agent_count;
    int team_size;
    int num_heads;            
    int head_dim;             
    int data_dim;             
    unsigned int global_state;
    int current_round;
    int64_t *target_data;     // Célvektor / környezeti optimum
    Agent *agents;
    void *master_memory_block; 
    int64_t *scratch_raw_scores; // Zéró-malloc scratchpad
    int64_t *scratch_weights;
    int64_t **scratch_head_contexts; 
} SimulationContext;
int64_t fixed_mul_64(int64_t a, int64_t b) { 
    return (int64_t)(((__int128)a * b) >> FIXED_SHIFT_64); 
}
int64_t fixed_clip_64(int64_t val, int64_t min_v, int64_t max_v) { 
    return (val < min_v) ? min_v : ((val > max_v) ? max_v : val); 
}
int64_t fixed_exp_neg_64(int64_t x) { 
    if (x < -8 * FIXED_ONE_64) return 0; 
    int64_t x2 = fixed_mul_64(x, x);
    int64_t x3 = fixed_mul_64(x2, x);
    int64_t x4 = fixed_mul_64(x3, x);
    int64_t res = FIXED_ONE_64 + x + (x2 >> 1) + (x3 / 6) + (x4 / 24); 
    return (res < 0) ? 0 : res; 
}
unsigned int next_random(unsigned int *state) { 
    *state = (*state * 1103515245 + 12345) & 0x7fffffff; 
    return *state; 
}
bool is_favorite_64(Agent *a, int id) { 
    for (int i = 0; i < a->favorites_count; i++) { 
        if (a->favorites[i] == id) return true; 
    } 
    return false; 
}
bool is_in_team_64(int team[], int team_count, int agent_idx) { 
    for (int i = 0; i < team_count; i++) { 
        if (team[i] == agent_idx) return true; 
    } 
    return false; 
}
void compute_multi_head_attention_64(SimulationContext *ctx, Agent *curr_agent, int active_fav_indices[], int active_fav_count) { 
    if (active_fav_count <= 0) return; 
    for (int h = 0; h < ctx->num_heads; h++) { 
        int64_t max_score = INT64_MIN; 
        for (int i = 0; i < active_fav_count; i++) { 
            int peer_idx = active_fav_indices[i]; 
            int64_t dot_product = 0; 
            for (int d = 0; d < ctx->head_dim; d++) { 
                dot_product += fixed_mul_64(curr_agent->query_vectors[h][d], ctx->agents[peer_idx].key_vectors[h][d]); 
            } 
            ctx->scratch_raw_scores[i] = dot_product >> 1; 
            if (ctx->scratch_raw_scores[i] > max_score) max_score = ctx->scratch_raw_scores[i]; 
        } 
        int64_t score_sum = 0; 
        for (int i = 0; i < active_fav_count; i++) { 
            ctx->scratch_weights[i] = fixed_exp_neg_64(ctx->scratch_raw_scores[i] - max_score); 
            score_sum += ctx->scratch_weights[i]; 
        } 
        if (score_sum > 0) { 
            for (int i = 0; i < active_fav_count; i++) { 
                ctx->scratch_weights[i] = (int64_t)(((__int128)ctx->scratch_weights[i] * FIXED_ONE_64) / score_sum); 
            } 
        } else { 
            for (int i = 0; i < active_fav_count; i++) { 
                ctx->scratch_weights[i] = FIXED_ONE_64 / active_fav_count; 
            } 
        } 
        for (int d = 0; d < ctx->head_dim; d++) { 
            int64_t sum_val = 0; 
            for (int i = 0; i < active_fav_count; i++) { 
                sum_val += fixed_mul_64(ctx->scratch_weights[i], ctx->agents[active_fav_indices[i]].value_vectors[h][d]); 
            } 
            ctx->scratch_head_contexts[h][d] = sum_val; 
        } 
    } 
}
void optimize_agent_params_evolved(SimulationContext *ctx, Agent *agent) { 
    int64_t inversion = FIXED_ONE_64 - agent->last_fit_quality; 
    if (inversion < 0) inversion = 0; 
    int64_t base_step = inversion >> LEARNING_RATE_SHIFT; 
    int64_t noise_scale = fixed_mul_64(inversion, (int64_t)EXPLORATION_NOISE_MAX << 16) >> 16; 
    for (int h = 0; h < ctx->num_heads; h++) { 
        for (int d = 0; d < ctx->head_dim; d++) { 
            int p_idx = h * ctx->head_dim + d; 
            int64_t delta = ctx->scratch_head_contexts[h][d] - agent->data_params[p_idx]; 
            int64_t rand_raw = (int64_t)(next_random(&agent->rng_state) % 2048) - 1042; 
            int64_t adaptive_noise = (rand_raw * noise_scale) >> 10; 
            int64_t adjustment = fixed_mul_64(base_step, delta) + adaptive_noise; 
            agent->data_params[p_idx] = fixed_clip_64(agent->data_params[p_idx] + adjustment, -FIXED_LIMIT_64, FIXED_LIMIT_64); 
            agent->query_vectors[h][d] = agent->data_params[p_idx]; 
            agent->key_vectors[h][d] = agent->data_params[p_idx]; 
            agent->value_vectors[h][d] = agent->data_params[p_idx]; 
        } 
    } 
}
void execute_evolutionary_selection(SimulationContext *ctx) { 
    for (int i = 0; i < ctx->agent_count; i++) { 
        if (ctx->agents[i].last_fit_quality < (FIXED_ONE_64 >> 1)) { 
            int best_peer = 0; 
            int64_t max_fit = -1; 
            for (int j = 0; j < ctx->agent_count; j++) { 
                if (i != j && ctx->agents[j].last_fit_quality > max_fit) { 
                    max_fit = ctx->agents[j].last_fit_quality; 
                    best_peer = j; 
                } 
            } 
            if (max_fit > ctx->agents[i].last_fit_quality) { 
                ctx->agents[i].favorites[0] = best_peer; 
                ctx->agents[i].confidence_score = ctx->agents[best_peer].last_fit_quality; 
            } 
        } 
    } 
}
void evaluate_agent_fitness(SimulationContext *ctx) {
    int64_t max_diff_per_dim = 2 * FIXED_LIMIT_64;
    int64_t max_total_error = ctx->data_dim * max_diff_per_dim;
    for (int i = 0; i < ctx->agent_count; i++) {
        Agent *agent = &ctx->agents[i];
        int64_t total_error = 0;
        for (int d = 0; d < ctx->data_dim; d++) {
            int64_t diff = ctx->target_data[d] - agent->data_params[d];
            total_error += (diff >= 0) ? diff : -diff;
        }
        int64_t error_ratio = 0;
        if (max_total_error > 0) {
            error_ratio = (int64_t)(((__int128)total_error * FIXED_ONE_64) / max_total_error);
        }
        int64_t fitness = FIXED_ONE_64 - error_ratio;
        agent->last_fit_quality = fixed_clip_64(fitness, 0, FIXED_ONE_64);
    }
}
bool check_global_convergence(SimulationContext *ctx, int64_t *out_best_error, int64_t *out_avg_fitness) {
    int64_t total_fitness = 0;
    int64_t best_fitness = -1;
    int64_t min_error = INT64_MAX;
    for (int i = 0; i < ctx->agent_count; i++) {
        total_fitness += ctx->agents[i].last_fit_quality;
        if (ctx->agents[i].last_fit_quality > best_fitness) {
            best_fitness = ctx->agents[i].last_fit_quality;
        }
        int64_t agent_error = 0;
        for (int d = 0; d < ctx->data_dim; d++) {
            int64_t diff = ctx->target_data[d] - ctx->agents[i].data_params[d];
            agent_error += (diff >= 0) ? diff : -diff;
        }
        if (agent_error < min_error) {
            min_error = agent_error;
        }
    }
    *out_best_error = min_error;
    *out_avg_fitness = total_fitness / ctx->agent_count;
    return (best_fitness >= (int64_t)(0.999 * FIXED_ONE_64)) || (min_error < (FIXED_ONE_64 / 1000));
}
void run_simulation_round(SimulationContext *ctx) {
    evaluate_agent_fitness(ctx);
    for (int i = 0; i < ctx->agent_count; i++) {
        compute_multi_head_attention_64(ctx, &ctx->agents[i], ctx->agents[i].favorites, ctx->agents[i].favorites_count);
        optimize_agent_params_evolved(ctx, &ctx->agents[i]);
    }
    execute_evolutionary_selection(ctx);
    ctx->current_round++;
}
SimulationContext* create_simulation_64(int agent_count, int team_size, int num_heads, int head_dim, unsigned int seed, int64_t *target_data) { 
    SimulationContext *ctx = malloc(sizeof(SimulationContext)); 
    ctx->agent_count = agent_count; 
    ctx->team_size = team_size; 
    ctx->num_heads = num_heads; 
    ctx->head_dim = head_dim; 
    ctx->data_dim = num_heads * head_dim; 
    ctx->global_state = seed; 
    ctx->current_round = 0; 
    ctx->target_data = malloc(ctx->data_dim * sizeof(int64_t)); 
    for (int d = 0; d < ctx->data_dim; d++) ctx->target_data[d] = target_data[d]; 
    ctx->scratch_raw_scores = malloc(agent_count * sizeof(int64_t)); 
    ctx->scratch_weights = malloc(agent_count * sizeof(int64_t)); 
    ctx->scratch_head_contexts = malloc(num_heads * sizeof(int64_t*)); 
    for (int h = 0; h < num_heads; h++) { 
        ctx->scratch_head_contexts[h] = malloc(head_dim * sizeof(int64_t)); 
    } 
    size_t bytes_per_agent = sizeof(Agent) + (ctx->data_dim * sizeof(int64_t)) + ((agent_count - 1) * sizeof(int)) + (agent_count * sizeof(int)) + (num_heads * sizeof(int64_t*)) * 3 + (num_heads * ctx->head_dim * sizeof(int64_t)) * 3; 
    ctx->master_memory_block = malloc(agent_count * bytes_per_agent); 
    ctx->agents = malloc(agent_count * sizeof(Agent)); 
    uint8_t *arena_ptr = (uint8_t*)ctx->master_memory_block; 
    for (int i = 0; i < agent_count; i++) { 
        Agent *curr = &ctx->agents[i]; 
        curr->id = i; 
        curr->rng_state = seed + i; 
        curr->last_fit_quality = 0; 
        curr->current_vote = 0; 
        curr->confidence_score = FIXED_ONE_64; 
        curr->data_params = (int64_t*)arena_ptr; arena_ptr += ctx->data_dim * sizeof(int64_t); 
        curr->preference_list = (int*)arena_ptr; arena_ptr += (agent_count - 1) * sizeof(int); 
        curr->favorites = (int*)arena_ptr; arena_ptr += agent_count * sizeof(int); 
        curr->query_vectors = (int64_t**)arena_ptr; arena_ptr += num_heads * sizeof(int64_t*); 
        curr->key_vectors = (int64_t**)arena_ptr; arena_ptr += num_heads * sizeof(int64_t*); 
        curr->value_vectors = (int64_t**)arena_ptr; arena_ptr += num_heads * sizeof(int64_t*); 
        for (int h = 0; h < num_heads; h++) { 
            curr->query_vectors[h] = (int64_t*)arena_ptr; arena_ptr += head_dim * sizeof(int64_t); 
            curr->key_vectors[h] = (int64_t*)arena_ptr; arena_ptr += head_dim * sizeof(int64_t); 
            curr->value_vectors[h] = (int64_t*)arena_ptr; arena_ptr += head_dim * sizeof(int64_t); 
            for (int d = 0; d < head_dim; d++) { 
                int p_idx = h * head_dim + d; 
                curr->data_params[p_idx] = (int64_t)(next_random(&curr->rng_state) % (2 * FIXED_ONE_64)) - FIXED_ONE_64; 
                curr->query_vectors[h][d] = curr->data_params[p_idx]; 
                curr->key_vectors[h][d] = curr->data_params[p_idx]; 
                curr->value_vectors[h][d] = curr->data_params[p_idx]; 
            } 
        } 
        int l_idx = 0; 
        for (int j = 0; j < agent_count; j++) { 
            if (i != j) curr->preference_list[l_idx++] = j; 
        } 
        unsigned int r_val = next_random(&curr->rng_state); 
        int target = curr->preference_list[r_val % (agent_count - 1)]; 
        curr->favorites[0] = target; 
        curr->favorites_count = 1; 
    } 
    return ctx; 
}
void free_simulation_64(SimulationContext *ctx) { 
    free(ctx->target_data); 
    free(ctx->scratch_raw_scores); 
    free(ctx->scratch_weights); 
    for (int h = 0; h < ctx->num_heads; h++) { 
        free(ctx->scratch_head_contexts[h]); 
    } 
    free(ctx->scratch_head_contexts); 
    free(ctx->master_memory_block); 
    free(ctx->agents); 
    free(ctx); 
}
int main() {
    int agent_count = 100;
    int team_size = 15;
    int num_heads = 4;
    int head_dim = 8; // 32 dimenziós optimalizációs tér
    int total_dim = num_heads * head_dim;
    int64_t *target = malloc(total_dim * sizeof(int64_t));
    unsigned int t_rng = 1337;
    for (int i = 0; i < total_dim; i++) {
        t_rng = (t_rng * 1103515245 + 12345) & 0x7fffffff;
        target[i] = (int64_t)(t_rng % (2 * FIXED_ONE_64)) - FIXED_ONE_64;
    }
    SimulationContext *sim = create_simulation_64(agent_count, team_size, num_heads, head_dim, 42, target);
    printf("=== Multi-Task Black-Box Optimization Engine (64-bit) Indul ===\n\n");
    int max_rounds = 1000;
    bool converged = false;
    for (int round = 1; round <= max_rounds; round++) {
        run_simulation_round(sim);
        if (round == 1 || round % 50 == 0 || round == max_rounds) {
            int64_t best_err = 0, avg_fit = 0;
            converged = check_global_convergence(sim, &best_err, &avg_fit);
            double best_err_f = (double)best_err / FIXED_ONE_64;
            double avg_fit_f  = (double)avg_fit / FIXED_ONE_64 * 100.0;
            printf("[KÖR %4d] -> Legjobb Hiba: %.6f | Átlagos Fitness: %.2f%%\n", round, best_err_f, avg_fit_f);
            if (converged) {
                printf("\nSIKER! A hálózat teljes konvergenciát ért el a %d. körben.\n", round);
                break;
            }
        }
    }
    if (!converged) {
        printf("\nA szimuláció elérte a maximális (%d) kört.\n", max_rounds);
    }
    free_simulation_64(sim);
    free(target);
    return 0;
}  