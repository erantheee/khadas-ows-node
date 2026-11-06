#include "khadas_ows/policy.hpp"
#include <cstdlib>
#include <iostream>
using namespace khadas_ows;
void check(bool v,const char* m){if(!v){std::cerr<<m<<'\n';std::exit(1);}}
int main(){
    ConversationPolicy p; PolicyInput in; in.external_speech=true; in.elapsed_ms=100;
    for(int i=0;i<4;++i) check(p.update(in).media_percent==100,"no early duck");
    auto preview=p.update(in); check(preview.state==ConversationState::anticipating&&preview.media_percent==70,"500 ms preview");
    for(int i=0;i<3;++i) p.update(in); auto engaged=p.update(in); check(engaged.media_percent==60,"800 ms confirmation");
    in.external_speech=false; auto hold=p.update(in); check(hold.state==ConversationState::holding&&hold.media_percent==60,"hold");
    in.elapsed_ms=1500; check(p.update(in).state==ConversationState::idle,"hold expires");
    p.reset(); in={true,false,true,true,true,1000}; check(p.update(in).state==ConversationState::idle,"fresh self identity blocks external action");
    p.reset(); in={true,false,false,false,false,1000}; check(p.update(in).state==ConversationState::idle,"invalid input fails closed");
    std::cout<<"policy_test passed\n";
}

