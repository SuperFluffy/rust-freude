// The `quote!` macro requires deep recursion.
#![recursion_limit = "192"]

extern crate proc_macro;
#[macro_use]
extern crate quote;
extern crate syn;

use proc_macro::TokenStream;
use syn::Ident;

mod internals;

use internals::{
    Container,
    Field,
};

struct IntegratorSettings<'a> {
    state: Field<'a>,
    stepper: Field<'a>,
    system: Field<'a>,
}

#[proc_macro_derive(Integrator, attributes(freude))]
pub fn derive_integrator(input: TokenStream) -> TokenStream {
    let ast = syn::parse_derive_input(&input.to_string()).unwrap();
    expand_derive_integrator(&ast).parse().unwrap()
}

fn expand_derive_integrator(input: &syn::DeriveInput) -> quote::Tokens {
    let cont = Container::from_ast(input);

    let ident = &cont.ident;

    let (impl_generics, ty_generics, where_clause) = cont.generics.split_for_impl();
    let dummy_const = Ident::new(format!("_IMPL_INTEGRATOR_FOR_{}", ident));

    let settings = IntegratorSettings::from_container(&cont);;

    let state_ident = settings.state.ident.clone();
    let state_type = settings.state.ty.clone();
    let stepper_ident = settings.stepper.ident.clone();
    let stepper_type = settings.stepper.ty.clone();
    let system_ident = settings.system.ident.clone();
    let system_type = settings.system.ty.clone();

    let after_step = if let Some(ref path) = cont.attrs.after_step {
        quote!{
            #path(self, dt);
        }
    } else {
        quote!{}
    };

    let after_integration = if let Some(ref path) = cont.attrs.after_integration {
        quote!{
            #path(self, dt, tacc);
        }
    } else {
        quote!{}
    };

    let impl_block = quote! {
        impl #impl_generics _freude::Integrator for #ident #ty_generics #where_clause {
            type State = #state_type;
            type Stepper = #stepper_type;
            type System = #system_type;

            fn integrate_n_steps(&mut self, n: usize) -> f64
            {
                let mut tacc = 0f64;

                let dt = self.#stepper_ident.timestep();

                // Ensure t is not exceeded
                for _ in 0..n {
                    self.#stepper_ident.do_step(&mut self.#system_ident, &mut self.#state_ident);
                    tacc += dt;
                    #after_step
                }

                #after_integration
                tacc
            }

            fn integrate_time(&mut self, t: f64) -> (f64, usize)
            {
                let mut tacc = 0f64;;
                let mut count = 0;

                let dt = self.#stepper_ident.timestep();

                // Ensure t is not exceeded
                while (tacc + dt) <= t {
                    self.#stepper_ident.do_step(&mut self.#system_ident, &mut self.#state_ident);
                    tacc += dt;
                    count += 1;
                    #after_step
                }

                #after_integration
                (tacc, count)
            }
        }
    };

    let generated = quote! {
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const #dummy_const: () = {
            extern crate freude as _freude;
            #impl_block
        };
    };
    generated
}

impl<'a> IntegratorSettings<'a> {
    fn from_container(cont: &'a Container) -> Self {
        let mut state = None;
        let mut stepper = None;
        let mut system = None;


        for field in &cont.fields {
            let mut field_is_used = false;
            if field.attrs.state {
                state = if state.is_none() {
                    if field_is_used {
                        panic!("Field '{}' can only be used in one role.", field.ident.as_ref().unwrap());
                    } else {
                        field_is_used = true;
                        Some(field.clone())
                    }
                } else {
                    panic!("Only one field can have '{}' attribute.", "state");
                };
            }

            if field.attrs.stepper {
                stepper = if stepper.is_none() {
                    if field_is_used {
                        panic!("Field '{}' can only be used in one role.", field.ident.as_ref().unwrap());
                    } else {
                        field_is_used = true;
                        Some(field.clone())
                    }
                } else {
                    panic!("Only one field can have '{}' attribute.", "stepper");
                };
            }

            if field.attrs.system {
                system = if system.is_none() {
                    if field_is_used {
                        panic!("Field '{}' can only be used in one role.", field.ident.as_ref().unwrap());
                    } else {
                        field_is_used = true;
                        Some(field.clone())
                    }
                } else {
                    panic!("Only one field can have '{}' attribute.", "system");
                };
            }
        }

        if state.is_none() || stepper.is_none() || system.is_none() {
            panic!("All integrator roles needs to be set.");
        }

        IntegratorSettings {
            state: state.unwrap(),
            stepper: stepper.unwrap(),
            system: system.unwrap(),
        }
    }
}
